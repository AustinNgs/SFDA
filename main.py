from easydl import *
from numpy.lib.function_base import kaiser
from csv2pic import *
from net import *
from lib import *
from tqdm import tqdm
import datetime
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.colors as col

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
sys.path[0] = '/home/lab-wu.shibin/dann'

#随机种子
seed_everything()
'''
#使用CPUorGPU
if args.misc.gpus < 1:
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    gpu_ids = []
    output_device = torch.device('cpu')
else:
    #返回可用GPU的list，选择list中第一个为训练GPU
    gpu_ids = select_GPUs(args.misc.gpus)
    output_device = gpu_ids[0]
'''
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
gpu_ids = [0]
output_device = gpu_ids

#记录起始时间
now = datetime.datetime.now().strftime('%b%d_%H-%M-%S')

#logging
log_dir = f'{args.log.root_dir}/{now}'

logger = SummaryWriter(log_dir)

#把头config文件中的配置写入并保存到logdir下，作为各自训练的参数
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

def visualize(source_feature: torch.Tensor, target_feature: torch.Tensor,
              filename: str, source_color='r', target_color='b'):
    """
    Visualize features from different domains using t-SNE.
    Args:
        source_feature (tensor): features from source domain in shape :math:`(minibatch, F)`
        target_feature (tensor): features from target domain in shape :math:`(minibatch, F)`
        filename (str): the file name to save t-SNE
        source_color (str): the color of the source features. Default: 'r'
        target_color (str): the color of the target features. Default: 'b'
    """
    source_feature = source_feature.numpy()
    target_feature = target_feature.numpy()
    features = np.concatenate([source_feature, target_feature], axis=0)

    # map features to 2-d using TSNE
    X_tsne = TSNE(n_components=2, random_state=33).fit_transform(features)

    # domain labels, 1 represents source while 0 represents target
    domains = np.concatenate((np.ones(len(source_feature)), np.zeros(len(target_feature))))

    # visualize using matplotlib
    plt.figure(figsize=(10, 10))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=domains, cmap=col.ListedColormap([target_color, source_color]), s=2)
    plt.savefig(filename)

def visualize2(source_low_feature: torch.Tensor, target_low_feature: torch.Tensor,
            source_high_feature: torch.Tensor, target_high_feature: torch.Tensor,
              filename: str, source_low_color='red',source_high_color='grey', target_low_color='blue',target_high_color='green'):
    """
    Visualize features from different domains using t-SNE.
    Args:
        source_feature (tensor): features from source domain in shape :math:`(minibatch, F)`
        target_feature (tensor): features from target domain in shape :math:`(minibatch, F)`
        filename (str): the file name to save t-SNE
        source_color (str): the color of the source features. Default: 'r'
        target_color (str): the color of the target features. Default: 'b'
    """
    source_low_feature = source_low_feature.numpy()
    source_high_feature = source_high_feature.numpy()
    target_low_feature = target_low_feature.numpy()
    target_high_feature = target_high_feature.numpy()
    features = np.concatenate([source_low_feature, source_high_feature,target_low_feature,target_high_feature], axis=0)

    # map features to 2-d using TSNE
    X_tsne = TSNE(n_components=2, random_state=10).fit_transform(features)

    # domain labels, 1 represents source while 0 represents target
    domains = np.concatenate((np.zeros(len(source_low_feature)),np.ones(len(source_high_feature)), 2*np.ones(len(target_low_feature)),3*np.ones(len(target_high_feature))))

    # visualize using matplotlib
    #map_marker = ('o','x','o','x')
    #markers = list(map(lambda x: map_marker[x],domains))
    plt.figure(figsize=(10, 10))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=domains, cmap=col.ListedColormap([target_low_color, target_high_color,source_low_color,source_high_color]), 
                #m = markers,
                s=2)
    plt.savefig(filename)

#模型字典，可选resnet与vgg
model_dict = {
    'resnet50': ResNet50Fc,
    'vgg16': VGG16Fc,
    'inception_v3': InceptionNet,
    'pnasnet5large':PnasNet,
    'wide_resnet101_2':WideResNet100Fc,
    'nasnetalarge':NasNet,
    'googlenet':GoogleNet
}

class TotalNet(nn.Module):
    def __init__(self):
        super(TotalNet, self).__init__()
        self.feature_extractor = model_dict[args.model.base_model](args.model.pretrained_model)
        classifier_output_dim = 2
        self.classifier = CLS(self.feature_extractor.output_num(), classifier_output_dim, bottle_neck_dim=2048)
        self.discriminator = AdversarialNetwork(2048,37) #36
        self.discriminator_separate = AdversarialNetwork(2048,2)

    def forward(self, x):
        f = self.feature_extractor(x)
        f, _, __, y = self.classifier(f)  #f是特征，y是概率
        d = self.discriminator(_)
        d_0 = self.discriminator_separate(_) #d,d_0是概率
        return y, d, d_0


totalNet = TotalNet()

#GPU分布式训练

feature_extractor = nn.DataParallel(totalNet.feature_extractor, device_ids=gpu_ids, output_device=output_device).train(True)
classifier = nn.DataParallel(totalNet.classifier, device_ids=gpu_ids, output_device=output_device).train(True)
discriminator = nn.DataParallel(totalNet.discriminator, device_ids=gpu_ids, output_device=output_device).train(True)
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
    discriminator.load_state_dict(data['discriminator'])
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
        low_im=[]
        high_im=[]
        for i, (im, label,idx) in enumerate(tqdm(source_train_dl, desc='source ')):
            im = im.cuda()
            label = label.cuda()
            for k in range(32):
                if label[k] == 0:
                    low_im.append(im[k])
                if label[k] == 1:
                    high_im.append(im[k])
            
            low_im_list = torch.tensor([item.cpu().detach().numpy() for item in low_im])#.cuda()
            high_im_list = torch.tensor([item.cpu().detach().numpy() for item in high_im])#.cuda()
            low_im_list =low_im_list.cuda()
            high_im_list=high_im_list.cuda()
            low_source_feature = feature_extractor.forward(low_im_list).cpu()
            high_source_feature = feature_extractor.forward(high_im_list).cpu()

            low_sourcefea.append(low_source_feature)
            high_sourcefea.append(high_source_feature)
            low_im=[]
            high_im=[]
        source_low_feature = torch.cat(low_sourcefea, dim=0)
        source_high_feature = torch.cat(high_sourcefea, dim=0)
        '''
            feature = feature_extractor.forward(im).cpu()
            _,test_sourceclsfea,_,_=classifier.forward(feature)
            test_sourceclsfea = test_sourceclsfea.cpu()
            _,test_sourcedisfea,_=discriminator_separate.forward(feature)
            test_sourcedisfea = test_sourcedisfea.cpu()
            all_sourcefea.append(feature)
            all_test_sourceclsfea.append(test_sourceclsfea)
            all_test_sourcedisfea.append(test_sourcedisfea)
            
        
        source_feature = torch.cat(all_sourcefea, dim=0)
        source_cls_feature = torch.cat(all_test_sourceclsfea,dim=0)
        source_dis_feature = torch.cat(all_test_sourcedisfea,dim=0)
        '''
        low_t_im=[]
        high_t_im=[]
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
            
            for k in range(32):
                if label[k] == 0:
                    low_number += 1
                    low_t_im.append(im[k])
                else:
                    high_number += 1
                    high_t_im.append(im[k])
            #print(low_number)
            #print(high_number)
            low_number=0
            high_number=0
            low_t_im_list = torch.tensor([item.cpu().detach().numpy() for item in low_t_im])#.cuda()
            high_t_im_list = torch.tensor([item.cpu().detach().numpy() for item in high_t_im])#.cuda()
            low_t_im_list =low_t_im_list.cuda()
            high_t_im_list=high_t_im_list.cuda()
            
            low_target_feature = feature_extractor.forward(low_t_im_list).cpu()
            high_target_feature = feature_extractor.forward(high_t_im_list).cpu()

            low_targetfea.append(low_target_feature)
            high_targetfea.append(high_target_feature)
            low_t_im=[]
            high_t_im=[]
            '''
            feature = feature_extractor.forward(im)
            feature1 = feature_extractor.forward(im).cpu()
            _,test_targetclsfea,_,_=classifier.forward(feature)
            test_targetclsfea = test_targetclsfea.cpu()
            _,test_targetdisfea,_=discriminator_separate.forward(feature)
            test_targetdisfea = test_targetdisfea.cpu()
            all_targetfea.append(feature1)
            all_test_targetclsfea.append(test_targetclsfea)
            all_test_targetdisfea.append(test_targetdisfea)
            

            feature, __, before_softmax, predict_prob = classifier.forward(feature) #标签分类器
            _,_,domain_prob = discriminator_separate.forward(__)   #公有域分类器

            for name in target_accumulator.names:
                globals()[name] = variable_to_numpy(globals()[name])

            target_accumulator.updateData(globals())
            '''
        target_low_feature = torch.cat(low_targetfea, dim=0)
        target_high_feature = torch.cat(high_targetfea, dim=0)
        '''
        target_feature = torch.cat(all_targetfea, dim=0)
        target_cls_feature = torch.cat(all_test_targetclsfea,dim=0)
        target_dis_feature = torch.cat(all_test_targetdisfea,dim=0)
        '''
    for x in target_accumulator:
        globals()[x] = target_accumulator[x]

    counters = [AccuracyCounter() for x in range(2 + 1)]

    for (each_predict_prob, each_label) in zip(predict_prob, label):
        counters[each_label].Ntotal += 1.0
        #最大概率的类别作为pred_id
        each_pred_id = np.argmax(each_predict_prob)
        #print(each_pred_id)
        #print(each_label)
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
    tSNE_filename = os.path.join(log_dir, 'TSNE.png')
    visualize2(source_low_feature, source_high_feature,target_low_feature, target_high_feature, tSNE_filename)
    '''    
    tSNE_filename = os.path.join(log_dir, 'TSNE.png')
    visualize(source_feature, target_feature, tSNE_filename)
    tSNE_filename1 = os.path.join(log_dir, 'TSNE_cls.png')
    visualize(source_cls_feature, target_cls_feature, tSNE_filename1)
    tSNE_filename2 = os.path.join(log_dir, 'TSNE_dis.png')
    visualize(source_dis_feature, target_dis_feature, tSNE_filename2)
    '''
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
optimizer_discriminator = OptimWithSheduler(
    optim.SGD(discriminator.parameters(), lr=args.train.lr , weight_decay=args.train.weight_decay, momentum=args.train.momentum, nesterov=True),
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
    #while global_step < args.train.min_step:
    #zip取source和target中len较小的一类作为iter进行train
    #iters = tqdm(zip(source_train_dl, target_train_dl), desc=f'epoch {epoch_id} ', total=max(len(source_train_dl), len(target_train_dl)))
    #epoch_id += 1

    for i in range(max(len(source_train_dl), len(target_train_dl))):
        im_source, label_source, idx_source = next(source_train_iter)
        im_target, label_target, idx_target = next(target_train_iter)
    #for i, ((im_source, label_source, idx_source), (im_target, label_target, idx_target)) in enumerate(iters):
        label_source = label_source.cuda()  #[0,1]
        idx_source = idx_source.cuda()  #[any]
        idx_target = idx_target.cuda() #3
        '''
        label_target = torch.zeros_like(label_target)#为什么要设置为0，label_source保持原有标签[0 or 1]
        '''
        #print(label_source)
        # =========================forward pass
        im_source = im_source.cuda()
        im_target = im_target.cuda()

        fc1_s = feature_extractor.forward(im_source)
        fc1_t = feature_extractor.forward(im_target)

        fc1_s, feature_source, fc2_s, predict_prob_source = classifier.forward(fc1_s)
        fc1_t, feature_target, fc2_t, predict_prob_target = classifier.forward(fc1_t)

        _,_,domain_prob_discriminator_source = discriminator.forward(feature_source)
        '''
        #domain_prob_discriminator_target = discriminator.forward(feature_target)
        #print(domain_prob_discriminator_source)
        '''
        _,_,domain_prob_discriminator_source_separate = discriminator_separate.forward(feature_source.detach())
        _,_,domain_prob_discriminator_target_separate = discriminator_separate.forward(feature_target.detach())
        '''
        #print(domain_prob_discriminator_source_separate.data.max(1, keepdim=False)[1])
        #print(domain_prob_discriminator_target_separate)
        '''
        # ==============================compute loss
        '''
        #adv_loss = torch.zeros(1, 1).cuda()
        '''
        adv_loss_separate = torch.zeros(1, 1).cuda()
        adv_loss_separate += nn.BCELoss()(domain_prob_discriminator_source_separate, torch.ones_like(domain_prob_discriminator_source_separate))*0.5
        adv_loss_separate += nn.BCELoss()(domain_prob_discriminator_target_separate, torch.zeros_like(domain_prob_discriminator_target_separate))*0.5
        '''
        #adv_loss_separate = nn.CrossEntropyLoss(reduction='none')(domain_prob_discriminator_target_separate, domain_prob_discriminator_source_separate.data.max(1, keepdim=False)[1])
        #adv_loss_separate = torch.mean(adv_loss_separate, dim=0, keepdim=True)
        '''
        # ============================== cross entropy loss
        #source data的标签分类loss, 定义为ce
        ce = nn.CrossEntropyLoss(reduction='none')(predict_prob_source, label_source)#*0.5
        ce = torch.mean(ce, dim=0, keepdim=True)#-0.29
        ce_new = F.nll_loss(F.log_softmax(predict_prob_source, dim=1), label_source)
        #源域混淆loss
        do = nn.CrossEntropyLoss(reduction='none')(domain_prob_discriminator_source, idx_source)#/32 #*0.1
        do = torch.mean(do, dim=0, keepdim=True)#*0.1
        do_new = F.nll_loss(F.log_softmax(domain_prob_discriminator_source, dim=1), idx_source)
        with OptimizerManager(
                [optimizer_finetune, optimizer_cls, optimizer_discriminator, optimizer_discriminator_separate]):
            loss = ce + 0.8*do + adv_loss_separate
            #loss = ce_new + do_new + adv_loss_separate
            #loss = ce + adv_loss_separate
            #loss = ce
            loss.backward()

        #global_step += 1
        #total_steps.update()
        
        running_loss1 += loss.item()
        running_loss2 += ce.item()
        running_loss3 += do.item()      
        running_loss4 += adv_loss_separate.item()

        #config中设置每隔10个iteration记录一次loss，acc
        #if epoch % args.log.log_interval == 0:
            #调用accuracycounter()
        counter = AccuracyCounter()
            #调用accuracycounter的其中一个方法计算mini-batch的acc
        counter.addOneBatch(variable_to_numpy(one_hot(label_source, 2)), variable_to_numpy(predict_prob_source))
            #调用accuracycounter的另一个方法基于addonebatch()计算whole batch的acc
        acc_train = torch.tensor([counter.reportAccuracy()]).cuda()
            #tensorboard画图
        logger.add_scalar('Loss/loss_source_da', running_loss3, epoch*max(len(source_train_dl), len(target_train_dl))+i)
        logger.add_scalar('Loss/loss_cls', running_loss2, epoch*max(len(source_train_dl), len(target_train_dl))+i)
        logger.add_scalar('Loss/loss_target_da', running_loss4,epoch*max(len(source_train_dl), len(target_train_dl))+i)
        logger.add_scalar('Loss/loss', running_loss1,epoch*max(len(source_train_dl), len(target_train_dl))+i)
        logger.add_scalar('Acc/acc_train', acc_train, epoch*max(len(source_train_dl), len(target_train_dl))+i)

        running_loss1 = 0.0
        running_loss2 = 0.0
        running_loss3 = 0.0
        running_loss4 = 0.0

        sys.stdout.write('\r epoch: %d, [iter: %d / all %d], loss_cls: %f, loss_source_da: %f, loss_target_da: %f' \
              % (epoch, i + 1, max(len(source_train_dl), len(target_train_dl)), ce.data.cpu().numpy(),
                 do.data.cpu().numpy(),adv_loss_separate.data.cpu().numpy().item()))
        sys.stdout.flush()
    #print('\n')
        #config中设置每隔5个epoch测试一次acc
    if epoch % args.test.test_interval == 0:
        counters = [AccuracyCounter() for x in range(2 + 1)]
        with TrainingModeManager([feature_extractor, classifier, discriminator_separate], train=False) as mgr, \
                Accumulator(['feature', 'predict_prob', 'label', 'domain_prob', 'before_softmax']) as target_accumulator, \
                torch.no_grad():

############################切换训练过程中看验证集还是测试集的结果#################################################
            
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
        #Testing_acc = acc_test.item()
        logger.add_scalar('Acc/acc_test',acc_test.item(), epoch)
        #clear_output()

        data = {
            "feature_extractor": feature_extractor.state_dict(),
            'classifier': classifier.state_dict(),
            'discriminator': discriminator.state_dict() if not isinstance(discriminator, Nonsense) else 1.0,
            'discriminator_separate': discriminator_separate.state_dict()
        }

        if acc_test > best_acc:
            best_acc = acc_test
            with open(join(log_dir, 'best.pkl'), 'wb') as f:
                torch.save(data, f)

        with open(join(log_dir, 'current.pkl'), 'wb') as f:
            torch.save(data, f)
'''
fc1_s.to_csv('./20211002/train_rawfea.csv')
fc2_s.to_csv('./20211002/train_dafea.csv')
fc1_t.to_csv('./20211002/test_rawfea.csv')
fc2_t.to_csv('./20211002/test_dafea.csv')
'''