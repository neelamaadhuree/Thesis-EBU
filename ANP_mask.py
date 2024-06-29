import os
import time
import argparse
import numpy as np
from collections import OrderedDict
import torch
import models


class ANPMask:

    def __init__(self, args,  model):
        self.model = model
        self.device = args.device
        self.args = args


    def mask(self, clean_val_loader, clean_test_loader, poison_test_loader):
        net = self.model
        args = self.args
        criterion = torch.nn.CrossEntropyLoss().to(self.device)
        parameters = list(net.named_parameters())
        mask_params = [v for n, v in parameters if "neuron_mask" in n]
        mask_optimizer = torch.optim.SGD(mask_params, lr=args.lr, momentum=0.9)
        noise_params = [v for n, v in parameters if "neuron_noise" in n]
        noise_optimizer = torch.optim.SGD(noise_params, lr=args.anp_eps / args.anp_steps)

        # Step 3: train backdoored models
        print('Iter \t lr \t Time \t TrainLoss \t TrainACC \t PoisonLoss \t PoisonACC \t CleanLoss \t CleanACC')
        nb_repeat = int(np.ceil(args.nb_iter / args.print_every))
        for i in range(nb_repeat):
            start = time.time()
            lr = mask_optimizer.param_groups[0]['lr']
            train_loss, train_acc = self.mask_train(model=net, criterion=criterion, data_loader=clean_val_loader,
                                            mask_opt=mask_optimizer, noise_opt=noise_optimizer)
            cl_test_loss, cl_test_acc = self.test(model=net, criterion=criterion, data_loader=clean_test_loader)
            po_test_loss, po_test_acc = self.test(model=net, criterion=criterion, data_loader=poison_test_loader)
            end = time.time()
            print('{} \t {:.3f} \t {:.1f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}'.format(
                (i + 1) * args.print_every, lr, end - start, train_loss, train_acc, po_test_loss, po_test_acc,
                cl_test_loss, cl_test_acc))
        self.save_mask_scores(net.state_dict(), os.path.join(args.anp_output_dir, 'mask_values.txt'))


    def load_state_dict(self, net, orig_state_dict):
        if "state_dict" in orig_state_dict.keys():
            orig_state_dict = orig_state_dict["state_dict"]

        new_state_dict = OrderedDict()
        for k, v in net.state_dict().items():
            if k in orig_state_dict.keys():
                new_state_dict[k] = orig_state_dict[k]
            elif 'running_mean_noisy' in k or 'running_var_noisy' in k or 'num_batches_tracked_noisy' in k:
                new_state_dict[k] = orig_state_dict[k[:-6]].clone().detach()
            else:
                new_state_dict[k] = v
        net.load_state_dict(new_state_dict)


    def clip_mask(self, model, lower=0.0, upper=1.0):
        params = [param for name, param in model.named_parameters() if 'neuron_mask' in name]
        with torch.no_grad():
            for param in params:
                param.clamp_(lower, upper)


    def sign_grad(self, model):
        noise = [param for name, param in model.named_parameters() if 'neuron_noise' in name]
        for p in noise:
            p.grad.data = torch.sign(p.grad.data)


    def perturb(self, model, is_perturbed=True):
        for name, module in model.named_modules():
            if isinstance(module, models.NoisyBatchNorm2d) or isinstance(module, models.NoisyBatchNorm1d):
                module.perturb(is_perturbed=is_perturbed)


    def include_noise(self, model):
        for name, module in model.named_modules():
            if isinstance(module, models.NoisyBatchNorm2d) or isinstance(module, models.NoisyBatchNorm1d):
                module.include_noise()


    def exclude_noise(self, model):
        for name, module in model.named_modules():
            if isinstance(module, models.NoisyBatchNorm2d) or isinstance(module, models.NoisyBatchNorm1d):
                module.exclude_noise()


    def reset(self, model, rand_init):
        for name, module in model.named_modules():
            if isinstance(module, models.NoisyBatchNorm2d) or isinstance(module, models.NoisyBatchNorm1d):
                module.reset(rand_init=rand_init, eps=self.args.anp_eps)


    def mask_train(self, model, criterion, mask_opt, noise_opt, data_loader):
        args = self.args
        model.train()
        total_correct = 0
        total_loss = 0.0
        nb_samples = 0
        for i, (images, labels, flag) in enumerate(data_loader):
            images, labels = images.to(self.device), labels.to(self.device)
            nb_samples += images.size(0)

            # step 1: calculate the adversarial perturbation for neurons
            if args.anp_eps > 0.0:
                self.reset(model, rand_init=True)
                for _ in range(args.anp_steps):
                    noise_opt.zero_grad()

                    self.include_noise(model)
                    output_noise = model(images)
                    loss_noise = - criterion(output_noise, labels)

                    loss_noise.backward()
                    self.sign_grad(model)
                    noise_opt.step()

            # step 2: calculate loss and update the mask values
            mask_opt.zero_grad()
            if args.anp_eps > 0.0:
                self.include_noise(model)
                output_noise = model(images)
                loss_rob = criterion(output_noise, labels)
            else:
                loss_rob = 0.0

            self.exclude_noise(model)
            output_clean = model(images)
            loss_nat = criterion(output_clean, labels)
            loss = args.anp_alpha * loss_nat + (1 - args.anp_alpha) * loss_rob

            pred = output_clean.data.max(1)[1]
            total_correct += pred.eq(labels.view_as(pred)).sum()
            total_loss += loss.item()
            loss.backward()
            mask_opt.step()
            self.clip_mask(model)

        loss = total_loss / len(data_loader)
        acc = float(total_correct) / nb_samples
        return loss, acc


    def test(self, model, criterion, data_loader):
        model.eval()
        total_correct = 0
        total_loss = 0.0
        with torch.no_grad():
            for i, (images, labels, gt_labels, isCleans) in enumerate(data_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                output = model(images)
                total_loss += criterion(output, labels).item()
                pred = output.data.max(1)[1]
                total_correct += pred.eq(labels.data.view_as(pred)).sum()
        loss = total_loss / len(data_loader)
        acc = float(total_correct) / len(data_loader.dataset)
        return loss, acc


    def save_mask_scores(self, state_dict, file_name):
        mask_values = []
        count = 0
        for name, param in state_dict.items():
            if 'neuron_mask' in name:
                for idx in range(param.size(0)):
                    neuron_name = '.'.join(name.split('.')[:-1])
                    mask_values.append('{} \t {} \t {} \t {:.4f} \n'.format(count, neuron_name, idx, param[idx].item()))
                    count += 1
        with open(file_name, "w") as f:
            f.write('No \t Layer Name \t Neuron Idx \t Mask Score \n')
            f.writelines(mask_values)

