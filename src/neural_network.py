import numpy
import pylab

from collections import Counter
from random import shuffle
from typing import Optional
from dataclasses import dataclass
from sklearn.metrics import auc, roc_curve

from torch import nn
from torch import optim
from torch import save as torch_save
from torch import load as torch_load
from torch import from_numpy as torch_from_numpy
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torch.autograd import Variable

from configuration import Configuration


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5) # Applies a 2D convolution over an input signal composed of several input planes
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d(p = 0.5) # Randomly zeroes whole channels of the input tensor
        self.fc1 = nn.Linear(3380, 50) # Applies a linear transformation to the incoming data: y=Ax+b
        self.fc2 = nn.Linear(50, 2)

    def forward(self, x):
        x = nn.functional.relu(nn.functional.max_pool2d(self.conv1(x), 2)) # Aplies a 2D max pooling over an input signal composed of several input planes
        x = nn.functional.relu(nn.functional.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 3380)
        x = nn.functional.relu(self.fc1(x)) # # Non-linear Activations: Applies the rectified linear unit function element-wise. ReLU(x)=max(0,x).
        x = nn.functional.dropout(x, training=self.training) # Randomly zeroes some elements of input tensor with probability p using samples from bernoulli distribution.
        x = self.fc2(x)
        return nn.functional.log_softmax(x, dim=1) # Applies a softmax followed by a logarithm: log(softmax(x))


def initialize_neural_network(stored_model_path: Optional[str], config: Configuration) -> Net:
    
    model: Net = torch_load(stored_model_path) if stored_model_path else Net()

    if config.cuda:
        model.cuda()
    
    return model


@dataclass
class Orchestrator:
    model: Net
    optimizer: optim.SGD
    config: Configuration

    def train(self, epoch: int, training_data_loader: DataLoader) -> list:
        
        avg_loss_1: list = []
        train_loss: int = 0
        
        self.model.train()
        
        for batch_idx, (data, target) in enumerate(training_data_loader):
            if self.config.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
    #        print(OrderedDict(Counter(target.data.cpu().numpy())))             
            self.optimizer.zero_grad() # Clears the gradients of all optimized Variables.
            output = self.model(data)
            loss = nn.functional.nll_loss(output, target) # The negative log likelihood loss
            train_loss += nn.functional.nll_loss(output, target, size_average=False).data[0]
            loss.backward() # Computes the gradient of current variable
            self.optimizer.step() # Updates the parameters. Called once the gradients are computed using backward().
            if batch_idx % self.config.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(training_data_loader.dataset),
                    100. * batch_idx / len(training_data_loader), loss.data[0]))
    
        train_loss /= len(training_data_loader.dataset)
        avg_loss_1.append(train_loss)

        return avg_loss_1

    def test(self, testing_data_loader: DataLoader) -> list:
        
        self.model.eval()
        
        test_loss: int = 0
        correct: int = 0
        
        all_scores: list  = []
        all_labels: list  = []
        pred_labels: list  = []
        
        avg_loss_2: list = []
        
        for data, target in testing_data_loader:
            if self.config.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data, volatile=True), Variable(target)
            output = self.model(data)
            test_loss += nn.functional.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            scores = output[:,1]
            scores_np = scores.data.cpu().numpy()
            labels_np = target.data.cpu().numpy()
            all_scores = numpy.append(all_scores, scores_np)
            all_labels = numpy.append(all_labels, labels_np)
            pred_labels = numpy.append(pred_labels, pred)
            correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()
    
        test_loss /= len(testing_data_loader.dataset)
        avg_loss_2.append(test_loss)
        
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}% LR: {})\n'.format(
                test_loss, correct, len(testing_data_loader.dataset),
                100. * correct / len(testing_data_loader.dataset), self.config.learning_rate))   
        
        return all_scores, all_labels, pred_labels


    def test_hard_negative(self, hard_negative_testing_data_loader: DataLoader) -> None:
        
        self.model.eval()
        
        test_loss: int = 0
        correct: int = 0
        
        hard_neg_scores: list = []
        hard_neg_labels: list = []
        hard_neg_pred: list = []
        avg_loss_2: list = []

        for data, target in hard_negative_testing_data_loader:
            if self.config.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data, volatile=True), Variable(target)
            output = self.model(data)
            test_loss += nn.functional.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            scores = output[:,1]
            scores_np = scores.data.cpu().numpy()
            labels_np = target.data.cpu().numpy()
            hard_neg_scores = numpy.append(hard_neg_scores, scores_np)
            hard_neg_labels = numpy.append(hard_neg_labels, labels_np)
            hard_neg_pred = numpy.append(hard_neg_pred, pred)
            correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()
    
        test_loss /= len(hard_negative_testing_data_loader.dataset)
        avg_loss_2.append(test_loss)
        print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss, correct, len(hard_negative_testing_data_loader.dataset),
                100. * correct / len(hard_negative_testing_data_loader.dataset)))   
        return hard_neg_scores, hard_neg_labels, hard_neg_pred

    def get_sensitivity_at_fp(self, false_positive_ratios, true_positive_ratios, false_positives):
        
        for index in range(len(false_positive_ratios)):
        
            if (false_positive_ratios[index]>=false_positives):
                return round(true_positive_ratios[index], 4)
        
        return 0

    # TODO: better name than "get_s"   
    def get_s(self, false_positive_ratios, true_positive_ratios, interval_min = -3, interval_max = 0):
        
        for index in range(len(false_positive_ratios)):
        
            if (false_positive_ratios[index]<=0):
                false_positive_ratios[index]=0.000000001
    
        fpr_logscale = pylab.log10(false_positive_ratios)
        fpr_logscale[pylab.isinf(fpr_logscale)] = -10
        
        samples_logscale = pylab.linspace(interval_min, interval_max, 30)
        
        tpr_interpolated = pylab.interp(samples_logscale, fpr_logscale, true_positive_ratios)
        
        # TODO: better name than "s"
        s = pylab.mean(tpr_interpolated)
        
        return samples_logscale, tpr_interpolated, s

    def run_training_testing(self, training_data, positive_samples, negative_samples, testing_data_loader, max_num_negatives = 80000):
        
        for epoch in range(1, self.config.epochs + 1):
            
            if epoch%10 == 1:
            
                shuffle(negative_samples)
                if max_num_negatives != 0:
                    negative_samples = negative_samples[:max_num_negatives]
            
                training_data.samples = positive_samples + negative_samples
            
                # Encode the difficulty in the label to determine the weighting
                for idx in range(len(training_data.samples)):
                    (path, label) = training_data.samples[idx]
            
                    if 'negatives' in path:
                        if 'difficult' in path:
                            label = 3
                        elif 'medium' in path:
                            label = 2
                        elif 'easy' in path:
                            label = 1
                    else:
                        label = 0
                
                    training_data.samples[idx] = (path, label)
                    
                labels = [label for (path, label) in training_data.samples]
                label_count = Counter(labels)
                #print(label_count) # for debugging
            
                weights = []
                for (path, label) in training_data.samples:
                    weight = 0
                    if label == 3:
                        weight = 3. / label_count[3]
                    elif label == 2:
                        weight = 2. / label_count[2]
                    elif label == 1:
                        weight = 1. / label_count[1]
                    elif label == 0:
                        weight = 6. / label_count[0]
                
                    weights += [weight]
            
                assert(len(weights) == len(training_data))
            
                # Decode the label again
                for idx in range(len(training_data.samples)):
                    (path, label) = training_data.samples[idx]
                    if label == 3 or label == 2 or label == 1: label = 1
                    elif label == 0: label = 0
                    training_data.samples[idx] = (path, label)
            
                weights = torch_from_numpy(numpy.array(weights).astype(numpy.float64))
                sampler = WeightedRandomSampler(weights, len(training_data))
                training_data_loader = DataLoader(training_data, batch_size=40, sampler = sampler)       
                                    
            avg_loss_1: list = self.train(epoch, training_data_loader)
            
            # TODO: orchestrator.test() returns all_scores, all_labels, pred_labels, maybe handle internally in orchestrator?
            [scores, labels, pred_labels] = self.test(testing_data_loader)

            false_positive_ratios, true_positive_ratios, thresholds = roc_curve(labels, scores, pos_label = 1)

            fp_levels = false_positive_ratios   
            sensitivity_values = []
            for false_positives in fp_levels:
                sensitivity_values.append( self.get_sensitivity_at_fp(false_positive_ratios, true_positive_ratios, false_positives) )
        
            roc_auc = auc(false_positive_ratios, true_positive_ratios)
            roc_log_fpr, roc_tpr, roc_s = self.get_s(false_positive_ratios, true_positive_ratios)

            #TODO: split off in separate storage function, load
            if (roc_auc > best_auc):
                best_auc = roc_auc
        #        best_roc_log_fpr = roc_log_fpr
        #        best_roc_tpr = roc_tpr
                best_roc_s = roc_s
                best_fpr = false_positive_ratios
                best_tpr = true_positive_ratios               
                torch_save(self.model,'/home/ananthi/model/myModel9_best.pt')
                numpy.save('/home/ananthi/data/scores_9.npy', scores)
                numpy.save('/home/ananthi/data/labels_9.npy', labels)
                numpy.save('/home/ananthi/data/pred_labels_9.npy', pred_labels)
                numpy.save('/home/ananthi/data/best_fpr_9.npy', best_fpr)
                numpy.save('/home/ananthi/data/best_tpr_9.npy', best_tpr)
            
            torch_save(self.model,'/home/ananthi/model/myModel9.pt')
            torch_save(self.model.state_dict(), '/home/ananthi/model/mytraining9.pt') 
