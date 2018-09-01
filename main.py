# this reprository is constructed from CAAE reprository
# for paper Learning face age progression 
# 

import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from dataloader import *
from misc import *
from models import *
import pickle
from makeLabel import *
import os

## boolean variable indicating whether cuda is available
use_cuda = torch.cuda.is_available()

makeDir()
moveFiles()

# loading Images()
dataloader = loadImgs()

## build model and use cuda if available
if use_cuda:
    netE = Encoder().cuda()
    netD_img = Dimg().cuda()
    netD_z  = Dz().cuda()
    netG = Generator().cuda()
else:
    netE = Encoder()
    netD_img = Dimg()
    netD_z  = Dz()
    netG = Generator()

## apply weight initialization
netE.apply(weights_init)
netD_img.apply(weights_init)
netD_z.apply(weights_init)
netG.apply(weights_init)

## build optimizer for each networks
optimizerE = optim.Adam(netE.parameters(),lr=0.0002,betas=(0.5,0.999))
optimizerD_z = optim.Adam(netD_z.parameters(),lr=0.0002,betas=(0.5,0.999))
optimizerD_img = optim.Adam(netD_img.parameters(),lr=0.0002,betas=(0.5,0.999))
optimizerG = optim.Adam(netG.parameters(),lr=0.0002,betas=(0.5,0.999))

## build criterions to calculate loss, and use cuda if available
if use_cuda:
    BCE = nn.BCELoss().cuda()
    L1  = nn.L1Loss().cuda()
    CE = nn.CrossEntropyLoss().cuda()
    MSE = nn.MSELoss().cuda()
else:
    BCE = nn.BCELoss()
    L1  = nn.L1Loss()
    CE = nn.CrossEntropyLoss()
    MSE = nn.MSELoss()

## fixed variables to regress / progress age
fixed_l = -torch.ones(80*10).view(80,10)


for i,l in enumerate(fixed_l):
    l[i//8] = 1

fixed_l_v = Variable(fixed_l)

if use_cuda:
    fixed_l_v = fixed_l_v.cuda()


outf='./result_tv_gender'

if os.path.exists(outf):
    os.mkdir(outf)
# iterations =50
niter=50

# iters for 50 epochs
for epoch in range(niter):
    # enumerates(dataLoader)
    for i,(img_data,img_label) in enumerate(dataloader):

        # make image variable and class variable
        #  img_data -->  

        #  previously age was encoded  and then decoded
        img_data_v = Variable(img_data)
        #  
        # age is stored
        img_age = img_label/2
        # gender is stored
        img_gender = img_label%2*2-1

        img_age_v = Variable(img_age).view(-1,1)
        img_gender_v = Variable(img_gender.float())


        if epoch == 0 and i == 0:
            # noise data is given 
            fixed_noise = img_data[:8].repeat(10,1,1,1)

            fixed_g = img_gender[:8].view(-1,1).repeat(10,1)


            fixed_img_v = Variable(fixed_noise)
            fixed_g_v = Variable(fixed_g)

            pickle.dump(fixed_noise,open("fixed_noise.p","wb"))

            if use_cuda:
                fixed_img_v = fixed_img_v.cuda()
                fixed_g_v = fixed_g_v.cuda()
        if use_cuda:
            img_data_v = img_data_v.cuda()
            img_age_v = img_age_v.cuda()
            img_gender_v = img_gender_v.cuda()

        # make one hot encoding version of label
        batchSize = img_data_v.size(0)
        
        # (img_age,batch_size=20,n_l=)
        age_ohe = one_hot(img_age,batchSize,n_l,use_cuda)

        # prior distribution z_star, real_label, fake_label
        z_star = Variable(torch.FloatTensor(batchSize*n_z).uniform_(-1,1)).view(batchSize,n_z)
        # real_label 
        real_label = Variable(torch.ones(batchSize).fill_(1)).view(-1,1)
        # fake_label
        fake_label = Variable(torch.ones(batchSize).fill_(0)).view(-1,1)

        if use_cuda:
            z_star, real_label, fake_label = z_star.cuda(),real_label.cuda(),fake_label.cuda()


        ## train Encoder and Generator with reconstruction loss
        netE.zero_grad()
        netG.zero_grad()

        # EG_loss 1. L1 reconstruction loss
        z = netE(img_data_v)
        # output is reconstruct
        reconst = netG(z,age_ohe,img_gender_v)
        #so L1 loss is constructed for output(reconst) and original image data(img_data_v)
        EG_L1_loss = L1(reconst,img_data_v)


        # EG_loss 2. GAN loss - image
        z = netE(img_data_v)
        # output of generator is reconst
        reconst = netG(z,age_ohe,img_gender_v)
        # discriminatoroutput is given as discrimninator_reconst is got from D_img()
        D_reconst,_ = netD_img(reconst,age_ohe.view(batchSize,n_l,1,1),img_gender_v.view(batchSize,1,1,1))
        # gan loss is given from bce 
        G_img_loss = BCE(D_reconst,real_label)



        ## EG_loss 3. GAN loss - z
        Dz_prior = netD_z(z_star)
        Dz = netD_z(z)
        Ez_loss = BCE(Dz,real_label)

        ## EG_loss 4. TV loss - G
        reconst = netG(z.detach(),age_ohe,img_gender_v)
        G_tv_loss = TV_LOSS(reconst)

        EG_loss = EG_L1_loss + 0.0001*G_img_loss + 0.01*Ez_loss + G_tv_loss
        EG_loss.backward()

        optimizerE.step()
        optimizerG.step()



        ## train netD_z with prior distribution U(-1,1)
        netD_z.zero_grad()
        Dz_prior = netD_z(z_star)
        Dz = netD_z(z.detach())

        Dz_loss = BCE(Dz_prior,real_label)+BCE(Dz,fake_label)
        Dz_loss.backward()
        optimizerD_z.step()



        ## train D_img with real images
        netD_img.zero_grad()
        D_img,D_clf = netD_img(img_data_v,age_ohe.view(batchSize,n_l,1,1),img_gender_v.view(batchSize,1,1,1))
        D_reconst,_ = netD_img(reconst.detach(),age_ohe.view(batchSize,n_l,1,1),img_gender_v.view(batchSize,1,1,1))

        D_loss = BCE(D_img,real_label)+BCE(D_reconst,fake_label)
        D_loss.backward()
        optimizerD_img.step()



    ## save fixed img for every 20 step
    fixed_z = netE(fixed_img_v)
    fixed_fake = netG(fixed_z,fixed_l_v,fixed_g_v)
    vutils.save_image(fixed_fake.data,
                '%s/reconst_epoch%03d.png' % (outf,epoch+1),
                normalize=True)

    ## checkpoint
    if epoch%10==0:
        torch.save(netE.state_dict(),"%s/netE_%03d.pth"%(outf,epoch+1))
        torch.save(netG.state_dict(),"%s/netG_%03d.pth"%(outf,epoch+1))
        torch.save(netD_img.state_dict(),"%s/netD_img_%03d.pth"%(outf,epoch+1))
        torch.save(netD_z.state_dict(),"%s/netD_z_%03d.pth"%(outf,epoch+1))


    msg1 = "epoch:{}, step:{}".format(epoch+1,i+1)
    msg2 = format("EG_L1_loss:%f"%(EG_L1_loss.data[0]),"<30")+"|"+format("G_img_loss:%f"%(G_img_loss.data[0]),"<30")
    msg5 = format("G_tv_loss:%f"%(G_tv_loss.data[0]),"<30")+"|"+"Ez_loss:%f"%(Ez_loss.data[0])
    msg3 = format("D_img:%f"%(D_img.mean().data[0]),"<30")+"|"+format("D_reconst:%f"%(D_reconst.mean().data[0]),"<30")\
    +"|"+format("D_loss:%f"%(D_loss.data[0]),"<30")
    msg4 = format("D_z:%f"%(Dz.mean().data[0]),"<30")+"|"+format("D_z_prior:%f"%(Dz_prior.mean().data[0]),"<30")\
    +"|"+format("Dz_loss:%f"%(Dz_loss.data[0]),"<30")

    print()
    print(msg1)
    print(msg2)
    print(msg5)
    print(msg3)
    print(msg4)
    print()
    print("-"*80)
