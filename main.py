from easydl import *
from numpy.lib.function_base import kaiser
from csv2pic import *
from net import *
from lib import *
from tqdm import tqdm
import datetime
import torch

#是否在notebok中运行
if is_in_notebook():
    from tqdm import tqdm_notebook as tqdm
from torch import optim
from tensorboardX import SummaryWriter
#from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
cudnn.deterministic = True

import sys
sys.path[0] = '/home/lab-wu.shibin/SFDA'

seed_everything()

os.environ["CUDA_VISIBLE_DEVICES"] = "5"
gpu_ids = [0]
output_device = gpu_ids

now = datetime.datetime.now().strftime('%b%d_%H-%M-%S')

#logging
log_dir = f'{args.log.root_dir}/{now}'

logger = SummaryWriter(log_dir)

with open(join(log_dir, 'config.yaml'), 'w') as f:
    f.write(yaml.dump(save_config))

class ForeverDataIterator:
    r"""A data iterator that will never stop producing data"""
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.iter = iter(self.data_loader)

    def __next__(self):
        try:
            data = next(self.iter)
        except StopIteration:
            self.iter = iter(self.data_loader)
            data = next(self.iter)
        return data

    def __len__(self):
        return len(self.data_loader)

model_dict = {
    'resnet50': ResNet50Fc,
    'vgg16': VGG16Fc
}

class TotalNet(nn.Module):
    def __init__(self):
        super(TotalNet, self).__init__()
        self.feature_extractor = model_dict[args.model.base_model](args.model.pretrained_model)
        classifier_output_dim = 2
        self.classifier = CLS(self.feature_extractor.output_num(), classifier_output_dim, bottle_neck_dim=2048)
        self.subjectfusion = AdversarialNetwork(2048,37)  #Modify according to different source subject Number
        self.discriminator_separate = AdversarialNetwork(2048,2)

    def forward(self, x):
        f = self.feature_extractor(x)
        f, _, __, y = self.classifier(f)  #f是特征，y是概率
        d = self.subjectfusion(_)
        d_0 = self.discriminator_separate(_) #d,d_0是概率
        return y, d, d_0


totalNet = TotalNet()

#GPU分布式训练

feature_extractor = nn.DataParallel(totalNet.feature_extractor, device_ids=gpu_ids, output_device=output_device).train(True)
classifier = nn.DataParallel(totalNet.classifier, device_ids=gpu_ids, output_device=output_device).train(True)
subjectfusion = nn.DataParallel(totalNet.subjectfusion, device_ids=gpu_ids, output_device=output_device).train(True)
discriminator_separate = nn.DataParallel(totalNet.discriminator_separate, device_ids=gpu_ids, output_device=output_device).train(True)

#测试目标域
if args.test.test_only:
    all_targetfea=[]
    all_sourcefea=[]
    all_test_sourceclsfea=[]
    all_test_sourcedisfea=[]
    all_test_targetclsfea=[]
    all_test_targetdisfea=[]
    low_sourcefea=[]
    high_sourcefea=[]
    low_targetfea=[]
    high_targetfea=[]
    #导入训练阶段的best.pkl
    assert os.path.exists(args.test.resume_file)
    data = torch.load(open(args.test.resume_file, 'rb'))
    feature_extractor.load_state_dict(data['feature_extractor'])
    classifier.load_state_dict(data['classifier'])
    subjectfusion.load_state_dict(data['subjectfusion'])
    discriminator_separate.load_state_dict(data['discriminator_separate'])
    number = 0
    high_risk_number = 0
    Sens_correct = 0
    low_risk_number = 0
    Spec_correct = 0
    #定义counters
    counters = [AccuracyCounter() for x in range(2 + 1)]
    with TrainingModeManager([feature_extractor, classifier, discriminator_separate], train=False) as mgr, \
            Accumulator(['feature', 'predict_prob', 'label', 'domain_prob', 'before_softmax']) as target_accumulator, \
            torch.no_grad():

        for i, (im, label,idx) in enumerate(tqdm(target_test_dl, desc='target ')):
            #im = im.to(output_device)
            #label = label.to(output_device)
            low_number=0
            high_number=0
            im = im.cuda()
            label = label.cuda()
            feature = feature_extractor.forward(im)
            feature, __, before_softmax, predict_prob = classifier.forward(feature) #标签分类器
            _,_,domain_prob = discriminator_separate.forward(__)   #公有域分类器

            for name in target_accumulator.names:
                globals()[name] = variable_to_numpy(globals()[name])

            target_accumulator.updateData(globals())
            
            
    for x in target_accumulator:
        globals()[x] = target_accumulator[x]

    counters = [AccuracyCounter() for x in range(2 + 1)]

    for (each_predict_prob, each_label) in zip(predict_prob, label):
        counters[each_label].Ntotal += 1.0
        each_pred_id = np.argmax(each_predict_prob)
        if each_label == 1:
            high_risk_number += 1
            if each_pred_id == each_label and each_label == 1:
                Sens_correct += 1

        if each_label == 0:
            low_risk_number += 1
            if each_pred_id == each_label and each_label == 0:
                Spec_correct += 1

        if each_pred_id == each_label:
            counters[each_label].Ncorrect += 1.0
            number+=1

    print(high_risk_number)
    print(Sens_correct)
    print(f'Sensitivity is {Sens_correct/high_risk_number}')
    print(low_risk_number)
    print(Spec_correct)
    print(f'Specitivity is {Spec_correct/low_risk_number}')
    acc_tests = [x.reportAccuracy() for x in counters if not np.isnan(x.reportAccuracy())]
    acc_test = torch.ones(1, 1) * np.mean(acc_tests)
    print(f'test accuracy is {acc_test.item()}')
    exit(0)

# ===================optimizer
#降低每个step学习率直到10000iteration
scheduler = lambda step, initial_lr: inverseDecaySheduler(step, initial_lr, gamma=10, power=0.75, max_iter=10000)
#优化器定义
optimizer_finetune = OptimWithSheduler(
    optim.SGD(feature_extractor.parameters(), lr=args.train.lr / 10.0, weight_decay=args.train.weight_decay, momentum=args.train.momentum, nesterov=True),
    scheduler)
optimizer_cls = OptimWithSheduler(
    optim.SGD(classifier.parameters(), lr=args.train.lr , weight_decay=args.train.weight_decay, momentum=args.train.momentum, nesterov=True),
    scheduler)
optimizer_subjectfusion = OptimWithSheduler(
    optim.SGD(subjectfusion.parameters(), lr=args.train.lr , weight_decay=args.train.weight_decay, momentum=args.train.momentum, nesterov=True),
    scheduler)
optimizer_discriminator_separate = OptimWithSheduler(
    optim.SGD(discriminator_separate.parameters(), lr=args.train.lr , weight_decay=args.train.weight_decay, momentum=args.train.momentum, nesterov=True),
    scheduler)

global_step = 0
best_acc = 0

#可扩展进度条
#total_steps = tqdm(range(args.train.min_step*args.train.iters_per_epoch),desc='global step')
epoch_id = 0

source_train_iter = ForeverDataIterator(source_train_dl)
target_train_iter = ForeverDataIterator(target_train_dl)

running_loss1 = 0.0
running_loss2 = 0.0
running_loss3 = 0.0
running_loss4 = 0.0
FC1_S=[]
FC2_S=[]
FC1_T=[]
FC2_T=[]
for epoch in range(args.train.min_step):

    for i in range(max(len(source_train_dl), len(target_train_dl))):
        im_source, label_source, idx_source = next(source_train_iter)
        im_target, label_target, idx_target = next(target_train_iter)
    
        label_source = label_source.cuda()  #[0,1]
        idx_source = idx_source.cuda()  #[any]
        idx_target = idx_target.cuda() #3

        # =========================forward pass
        im_source = im_source.cuda()
        im_target = im_target.cuda()

        fc1_s = feature_extractor.forward(im_source)
        fc1_t = feature_extractor.forward(im_target)

        fc1_s, feature_source, fc2_s, predict_prob_source = classifier.forward(fc1_s)
        fc1_t, feature_target, fc2_t, predict_prob_target = classifier.forward(fc1_t)

        _,_,domain_prob_discriminator_source = subjectfusion.forward(feature_source)
        _,_,domain_prob_discriminator_source_separate = discriminator_separate.forward(feature_source.detach())
        _,_,domain_prob_discriminator_target_separate = discriminator_separate.forward(feature_target.detach())

        # ==============================compute loss
        #DA loss
        adv_loss_separate = torch.zeros(1, 1).cuda()
        adv_loss_separate += nn.BCELoss()(domain_prob_discriminator_source_separate, torch.ones_like(domain_prob_discriminator_source_separate))*0.5
        adv_loss_separate += nn.BCELoss()(domain_prob_discriminator_target_separate, torch.zeros_like(domain_prob_discriminator_target_separate))*0.5

        #lable classification loss
        ce = nn.CrossEntropyLoss(reduction='none')(predict_prob_source, label_source)#*0.5
        ce = torch.mean(ce, dim=0, keepdim=True)
        
        #source fusion loss
        do = nn.CrossEntropyLoss(reduction='none')(domain_prob_discriminator_source, idx_source)
        do = torch.mean(do, dim=0, keepdim=True)#*0.1
        
        with OptimizerManager(
                [optimizer_finetune, optimizer_cls, optimizer_subjectfusion, optimizer_discriminator_separate]):
            #Explore optimal weight of SF and DA here
            loss = ce + 1*do + 1*adv_loss_separate  #Manually change weight
            loss.backward()

        running_loss1 += loss.item()
        running_loss2 += ce.item()
        running_loss3 += do.item()      
        running_loss4 += adv_loss_separate.item()

        counter = AccuracyCounter()
        counter.addOneBatch(variable_to_numpy(one_hot(label_source, 2)), variable_to_numpy(predict_prob_source))
        acc_train = torch.tensor([counter.reportAccuracy()]).cuda()
        #tensorboard drawing
        logger.add_scalar('Loss/loss_source_da', running_loss3, epoch*max(len(source_train_dl), len(target_train_dl))+i)
        logger.add_scalar('Loss/loss_cls', running_loss2, epoch*max(len(source_train_dl), len(target_train_dl))+i)
        logger.add_scalar('Loss/loss_target_da', running_loss4,epoch*max(len(source_train_dl), len(target_train_dl))+i)
        logger.add_scalar('Loss/loss', running_loss1,epoch*max(len(source_train_dl), len(target_train_dl))+i)
        logger.add_scalar('Acc/acc_train', acc_train, epoch*max(len(source_train_dl), len(target_train_dl))+i)

        running_loss1 = 0.0
        running_loss2 = 0.0
        running_loss3 = 0.0
        running_loss4 = 0.0

        sys.stdout.write('\r epoch: %d, [iter: %d / all %d], loss_cls: %f, loss_source_fusion: %f, loss_adaptation: %f' \
              % (epoch, i + 1, max(len(source_train_dl), len(target_train_dl)), ce.data.cpu().numpy(),
                 do.data.cpu().numpy(),adv_loss_separate.data.cpu().numpy().item()))
        sys.stdout.flush()

    if epoch % args.test.test_interval == 0:
        counters = [AccuracyCounter() for x in range(2 + 1)]
        with TrainingModeManager([feature_extractor, classifier, discriminator_separate], train=False) as mgr, \
                Accumulator(['feature', 'predict_prob', 'label', 'domain_prob', 'before_softmax']) as target_accumulator, \
                torch.no_grad():
         
            for i, (im, label, idx) in enumerate(tqdm(target_train_dl, desc='testing ')):  
                im = im.cuda()
                label = label.cuda()
                idx = idx.cuda()

                feature = feature_extractor.forward(im)
                feature, __, before_softmax, predict_prob = classifier.forward(feature)
                _,_,domain_prob = discriminator_separate.forward(__)
                co = nn.CrossEntropyLoss(reduction='none')(predict_prob, label)
                co = torch.mean(co, dim=0, keepdim=True)

                for name in target_accumulator.names:
                    globals()[name] = variable_to_numpy(globals()[name])

                target_accumulator.updateData(globals())
        logger.add_scalar('Loss/val_loss', co.item(), epoch)
        for x in target_accumulator:
            globals()[x] = target_accumulator[x]
        

        counters = [AccuracyCounter() for x in range(2 + 1)]

        for (each_predict_prob, each_label) in zip(predict_prob, label):
            counters[each_label].Ntotal += 1.0
            each_pred_id = np.argmax(each_predict_prob)
            if each_pred_id == each_label:
                counters[each_label].Ncorrect += 1.0

        acc_tests = [x.reportAccuracy() for x in counters if not np.isnan(x.reportAccuracy())]
        acc_test = torch.ones(1, 1) * np.mean(acc_tests)
        print(acc_test)
        logger.add_scalar('Acc/acc_test',acc_test.item(), epoch)

        data = {
            "feature_extractor": feature_extractor.state_dict(),
            'classifier': classifier.state_dict(),
            'subjectfusion': subjectfusion.state_dict() if not isinstance(subjectfusion, Nonsense) else 1.0,
            'discriminator_separate': discriminator_separate.state_dict()
        }

        if acc_test > best_acc:
            best_acc = acc_test
            with open(join(log_dir, 'best.pkl'), 'wb') as f:
                torch.save(data, f)

        with open(join(log_dir, 'current.pkl'), 'wb') as f:
            torch.save(data, f)