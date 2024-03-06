# below: kwargs = {'num_workers': 18, 'pin_memory': True} if config.cuda else {}
#epochs = 50
#batch_size = 20
#learning_rate = 1e-3
#momentum = 0.8
#log_interval = 6

# below: unsharp_mask = preprocessing.UnsharpMaskCV2()
#data_train = ImageFolder(root='/home/ananthi/train/', transform=transforms.Compose([
#transforms.Lambda(lambda tensor:unsharp([tensor])),
#transforms.Lambda(lambda tensor:min_max_normalization(tensor / 15, 0, 255)),
#transforms.ToTensor(),
#transforms.ToPILImage(mode = 'L'),
#transforms.RandomHorizontalFlip(),
#transforms.RandomVerticalFlip(),
#transforms.RandomRotation(90),
#transforms.CenterCrop(65),
#transforms.ToTensor()
#]))   
##train_loader = DataLoader(data_train, batch_size=args.batch_size, shuffle=True, **kwargs)
#
#class_sample_count = [700, 1000] # dataset has 1681 class-0 samples, 681 class-1 samples, etc.
#dataset_length = 1700
#prob = 1 / torch.Tensor(class_sample_count)
#weight1 = prob[0]*np.ones(700)
#weight2 = prob[1]*np.ones(1000)
#weights = np.concatenate((weight1,weight2))
#weights = torch.from_numpy(weights)
#sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, dataset_length)
#train_loader = DataLoader(data_train, batch_size=args.batch_size, sampler = sampler, **kwargs)

# below: get_s()
#def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
#    torch.save(state, filename)
#    if is_best:
#        shutil.copyfile(filename, 'model_best.pth.tar')   
#
#if args.pretrained:
#    print("=> using pre-trained model '{}'".format(args.arch))
#    model = models.__dict__[args.arch](pretrained=True)
#else:
#    print("=> creating model '{}'".format(args.arch))
#    model = models.__dict__[args.arch]()
   
#model = Net()
#model.cuda()
#
#optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=1e-5)

#if args.resume:
#    if os.path.isfile(args.resume):
#            print("=> loading checkpoint '{}'".format(args.resume))
#            checkpoint = torch.load(args.resume)
#            args.start_epoch = checkpoint['epoch']
#            best_auc = checkpoint['best_auc']
#            model.load_state_dict(checkpoint['state_dict'])
#            optimizer.load_state_dict(checkpoint['optimizer'])
#            print("=> loaded checkpoint '{}' (epoch {})"
#                  .format(args.resume, checkpoint['epoch']))
#    else:
#        print("=> no checkpoint found at '{}'".format(args.resume))
   
# def create_hard_negatives_folder(info):
#     src_path = '/home/ananthi/normal_2/siemens/0/'
#     dest1 = '/home/ananthi/difficult/'
#     dest2 = '/home/ananthi/medium/'
#     dest3 = '/home/ananthi/easy/'
#     transposed_info = info.transpose()
#     sorted_info = transposed_info[transposed_info[:,1].argsort()]
# #    for i in range(len(sorted_info)):
# #        filename = str(int(sorted_info[i][0]))+ ".png"
# #        src_file = src_path + filename
# #        if (sorted_info[i][2] != sorted_info[i][3]):
# #            shutil.copy2(src_file, dest1)
# #        elif (sorted_info[i][1] <= -0.1):
# #            shutil.copy2(src_file, dest2)
# #        else:
# #            shutil.copy2(src_file, dest3)
           
           
#     l1 = int(0.1*len(sorted_info))
#     l2 = int(0.5*len(sorted_info))
#     l3 = len(sorted_info)
# #    filename = str(int(sorted_info[i][0]))+ ".png"
# #    src_file = src_path + filename       
#     for i in range(0, l1):
#         filename = str(int(sorted_info[i][0]))+ ".png"
#         src_file = src_path + filename
#         shutil.copy2(src_file, dest1)
#     for i in range(l1, l2):
#         filename = str(int(sorted_info[i][0]))+ ".png"
#         src_file = src_path + filename       
#         shutil.copy2(src_file, dest2)
#     for i in range(l2, l3):
#         filename = str(int(sorted_info[i][0]))+ ".png"
#         src_file = src_path + filename       
#         shutil.copy2(src_file, dest3)

# below: for epoch in range(1, config.epochs + 1):
#    if epoch <= 20:
#        lr = 0.01
#    elif 20 < epoch <= 250:
#        lr = 0.001
#    elif 250 < epoch <= 350:
#        lr = 0.0001           


# below:     torch.save(model.state_dict(), '/home/ananthi/model/mytraining9.pt') 
       
#    model.save_state_dict('/home/ananthi/model/mytraining.pt')
#    save_checkpoint({
#        'epoch': epoch + 1,
#        'arch': args.arch,
#        'state_dict': model.state_dict(),
#        'best_prec1': best_auc,
#        'optimizer' : optimizer.state_dict(),
#        }, is_best)  

# hard negative mining

#for epoch in range(1, args.epochs_validate + 1):  
#
#    [hard_neg_scores, hard_neg_labels, hard_neg_pred] = test_hard_neg(epoch)
#   
#index = np.arange(163000)
#info = np.array([index, hard_neg_scores, hard_neg_labels, hard_neg_pred])
#create_hard_negatives_folder(info)
          
# below: plt.xlabel('partial_fpr_log')
# below: plt.ylabel('tpr')
#plt.figure()
#plt.plot(best_roc_log_fpr,best_roc_tpr)
#plt.xlabel('partial_fpr_log_2')
#plt.ylabel('tpr')

#p_auc = auc(best_roc_log_fpr,best_roc_tpr)