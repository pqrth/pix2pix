-- usage example: DATA_ROOT=/path/to/data/ which_direction=BtoA name=expt1 th train.lua 
--
-- code derived from https://github.com/soumith/dcgan.torch
--

require 'torch'
require 'nn'
require 'optim'
util = paths.dofile('util/util.lua')
require 'image'
require 'models'
require 'warp2d'
require 'os'

opt = {
   DATA_ROOT = '',         -- path to images (should have subfolders 'train', 'val', etc)
   batchSize = 1,          -- # images in batch
   loadSize = 286,         -- scale images to this size
   fineSize = 256,         --  then crop to this size
   ngf = 64,               -- #  of gen filters in first conv layer
   ndf = 64,               -- #  of discrim filters in first conv layer
   input_nc = 3,           -- #  of input image channels
   output_nc = 3,          -- #  of output image channels
   niter = 200,            -- #  of iter at starting learning rate
   lr = 0.0002,            -- initial learning rate for adam
   beta1 = 0.5,            -- momentum term of adam
   ntrain = math.huge,     -- #  of examples per epoch. math.huge for full dataset
   flip = 1,               -- if flip the images for data argumentation
   display = 1,            -- display samples while training. 0 = false
   display_id = 10,        -- display window id.
   display_plot = 'errL1',    -- which loss values to plot over time. Accepted values include a comma seperated list of: errL1, errG, and errD
   gpu = 1,                -- gpu = 0 is CPU mode. gpu=X is GPU mode on GPU X
   name = '',              -- name of the experiment, should generally be passed on the command line
   which_direction = 'AtoB',    -- AtoB or BtoA
   phase = 'train',             -- train, val, test, etc
   preprocess = 'regular',      -- for special purpose preprocessing, e.g., for colorization, change this (selects preprocessing functions in util.lua)
   nThreads = 2,                -- # threads for loading data
   save_epoch_freq = 50,        -- save a model every save_epoch_freq epochs (does not overwrite previously saved models)
   save_latest_freq = 5000,     -- save the latest model every latest_freq sgd iterations (overwrites the previous latest model)
   print_freq = 50,             -- print the debug information every print_freq iterations
   display_freq = 100,          -- display the current results every display_freq iterations
   save_display_freq = 5000,    -- save the current display of results every save_display_freq_iterations
   continue_train=0,            -- if continue training, load the latest model: 1: true, 0: false
   serial_batches = 0,          -- if 1, takes images in order to make batches, otherwise takes them randomly
   serial_batch_iter = 1,       -- iter into serial image list
   checkpoints_dir = './checkpoints', -- models are saved here
   cudnn = 1,                         -- set to 0 to not use cudnn
   condition_GAN = 1,                 -- set to 0 to use unconditional discriminator
   use_GAN = 0,                       -- set to 0 to turn off GAN term
   use_L1 = 1,                        -- set to 0 to turn off L1 term
   which_model_netD = 'basic', -- selects model to use for netD
   which_model_netG = 'unet',  -- selects model to use for netG
   n_layers_D = 0,             -- only used if which_model_netD=='n_layers'
   lambda = 100,               -- weight on L1 term in objective
   useTPS = 0,
   nScales = 3,
   pad = 0,                    -- set non zero to pad by that much margin
   useSobel = 0, 
}

-- one-line argument parser. parses enviroment variables to override the defaults
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)

local input_nc = opt.input_nc
local output_nc = opt.output_nc
-- translation direction
local idx_A = nil
local idx_B = nil

if opt.which_direction=='AtoB' then
    idx_A = {1, input_nc}
    idx_B = {input_nc+1, input_nc+output_nc}
elseif opt.which_direction=='BtoA' then
    idx_A = {input_nc+1, input_nc+output_nc}
    idx_B = {1, input_nc}
else
    error(string.format('bad direction %s',opt.which_direction))
end

if opt.display == 0 then opt.display = false end

opt.manualSeed = torch.random(1, 10000) -- fix seed
print("Random Seed: " .. opt.manualSeed)
torch.manualSeed(opt.manualSeed)
torch.setdefaulttensortype('torch.FloatTensor')

-- create data loader
local data_loader = paths.dofile('data/data.lua')
print('#threads...' .. opt.nThreads)
local data = data_loader.new(opt.nThreads, opt)
print("Dataset Size: ", data:size())
tmp_d, tmp_paths = data:getBatch()

----------------------------------------------------------------------------
local function weights_init(m)
   local name = torch.type(m)
   if name:find('Convolution') then
      m.weight:normal(0.0, 0.02)
      m.bias:fill(0)
   elseif name:find('BatchNormalization') then
      if m.weight then m.weight:normal(1.0, 0.02) end
      if m.bias then m.bias:fill(0) end
   end
end


local ndf = opt.ndf
local ngf = opt.ngf
local real_label = 1
local fake_label = 0

function defineG(input_nc, output_nc, ngf)
    local netG = nil
    if     opt.which_model_netG == "encoder_decoder" then netG = defineG_encoder_decoder(input_nc, output_nc, ngf)
    elseif opt.which_model_netG == "unet" then netG = dewarp_multiscale(input_nc, output_nc, ngf,opt.fineSize+opt.pad*2,opt.fineSize+opt.pad*2,opt.nScales)
    elseif opt.which_model_netG == "unet_128" then netG = defineG_unet_128(input_nc, output_nc, ngf)
    else error("unsupported netG model")
    end
   
    netG:apply(weights_init)
  
    return netG
end

netLoss = define_loss(opt.fineSize+opt.pad*2,opt.fineSize+opt.pad*2)
netLoss:apply(weights_init)

function defineD(input_nc, output_nc, ndf)
    local netD = nil
    if opt.condition_GAN==1 then
        input_nc_tmp = input_nc
    else
        input_nc_tmp = 0 -- only penalizes structure in output channels
    end
    
    if     opt.which_model_netD == "basic" then netD = defineD_basic(input_nc_tmp, output_nc, ndf)
    elseif opt.which_model_netD == "n_layers" then netD = defineD_n_layers(input_nc_tmp, output_nc, ndf, opt.n_layers_D)
    else error("unsupported netD model")
    end
    
    netD:apply(weights_init)
    
    return netD
end


-- load saved models and finetune
if opt.continue_train == 1 then
   print('loading previously trained netG...')
   netG = util.load(paths.concat(opt.checkpoints_dir, opt.name, 'latest_net_G.t7'), opt)
   print('loading previously trained netD...')
   netD = util.load(paths.concat(opt.checkpoints_dir, opt.name, 'latest_net_D.t7'), opt)
else
  print('define model netG...')
  netG = defineG(input_nc, output_nc, ngf)
  print('define model netD...')
  netD = defineD(input_nc, output_nc, ndf)
end

print(netG)
print(netD)


local criterion = nn.BCECriterion()
local criterionAE = nn.AbsCriterion()
local criterionSobel = nn.AbsCriterion()
local criterionAEby2 = nn.AbsCriterion()
local criterionAEby4 = nn.AbsCriterion()
---------------------------------------------------------------------------
optimStateG = {
   learningRate = opt.lr,
   beta1 = opt.beta1,
}
optimStateD = {
   learningRate = opt.lr,
   beta1 = opt.beta1,
}
----------------------------------------------------------------------------
pad = opt.pad
local real_A = torch.Tensor(opt.batchSize, input_nc, opt.fineSize+pad*2, opt.fineSize+pad*2):fill(0)
local real_B = torch.Tensor(opt.batchSize, output_nc, opt.fineSize+pad*2, opt.fineSize+pad*2):fill(0)
local fake_B = torch.Tensor(opt.batchSize, output_nc, opt.fineSize+pad*2, opt.fineSize+pad*2):fill(0)
local fake_offsetSobel = torch.Tensor(opt.batchSize, 2, opt.fineSize+pad*2, opt.fineSize+pad*2):fill(0)
local fake_offsets = torch.Tensor(opt.batchSize, 2, opt.fineSize+pad*2, opt.fineSize+pad*2):fill(0)
local fake_offsetsBy2 = torch.Tensor(opt.batchSize, 2, (opt.fineSize+pad*2)/2, (opt.fineSize+pad*2)/2):fill(0)
local fake_offsetsBy4 = torch.Tensor(opt.batchSize, 2, (opt.fineSize+pad*2)/4, (opt.fineSize+pad*2)/4):fill(0)
local real_AB = torch.Tensor(opt.batchSize, output_nc + input_nc*opt.condition_GAN, opt.fineSize+pad*2, opt.fineSize+pad*2):fill(0)
local fake_AB = torch.Tensor(opt.batchSize, output_nc + input_nc*opt.condition_GAN, opt.fineSize+pad*2, opt.fineSize+pad*2):fill(0)
local errD, errG, errL1, errSobel = 0, 0, 0, 0
local epoch_tm = torch.Timer()
local tm = torch.Timer()
local data_tm = torch.Timer()
----------------------------------------------------------------------------

if opt.gpu > 0 then
   print('transferring to gpu...')
   require 'cunn'
   cutorch.setDevice(opt.gpu)
   real_A = real_A:cuda();
   real_B = real_B:cuda(); fake_B = fake_B:cuda();
   real_AB = real_AB:cuda(); fake_AB = fake_AB:cuda();
   fake_offsetSobel:cuda(); fake_offsets:cuda(); fake_offsetsBy2:cuda(); fake_offsetsBy4:cuda();
   if opt.cudnn==1 then
      netG = util.cudnn(netG); netD = util.cudnn(netD); netLoss = util.cudnn(netLoss);
   end
   netLoss:cuda();
   netD:cuda(); netG:cuda(); criterion:cuda(); criterionAE:cuda(); criterionSobel:cuda(); criterionAEby2:cuda(); criterionAEby4:cuda();
   print('done')
else
	print('running model on CPU')
end


local parametersD, gradParametersD = netD:getParameters()
local parametersG, gradParametersG = netG:getParameters()



if opt.display then disp = require 'display' end


function createRealFake()
--    print('enter createrealfake')
    -- load real
    data_tm:reset(); data_tm:resume()
    local real_data, data_path = data:getBatch()
    data_tm:stop()

    --real_A:copy(real_data[{ {}, idx_A, {}, {} }])
    real_A:fill(0)
    real_B:fill(0)
    if opt.useTPS == 0 then
	real_A[{{},{},{pad+1,pad+opt.fineSize},{pad+1,pad+opt.fineSize}}] = real_data[{ {}, idx_A, {}, {} }]:clone()
	real_B[{{},{},{pad+1,pad+opt.fineSize},{pad+1,pad+opt.fineSize}}] = real_data[{ {}, idx_B, {}, {} }]:clone()
    else
	real_A[{{},{},{pad+1,pad+opt.fineSize},{pad+1,pad+opt.fineSize}}] = real_data[{ {}, idx_B, {}, {} }]:clone()
    	real_B[{{},{},{pad+1,pad+opt.fineSize},{pad+1,pad+opt.fineSize}}] = real_data[{ {}, idx_B, {}, {} }]:clone()

    	-- artificial warping (data augmentation)
        for ind = 1, real_A:size(1) do
            local im = torch.Tensor(real_A[ind]:size()):copy(real_A[ind])
            img1 = torch.zeros(im:size())
            scaleSize = torch.floor(im:size()[2]*torch.uniform(0.6,1))
            padng = torch.floor((im:size()[2]-scaleSize)/2)
            im = image.scale(im, scaleSize,'bilinear')
            img1[{{},{padng+1,padng+scaleSize},{padng+1,padng+scaleSize}}] = im[{{},{},{}}]
            real_B[ind]:copy(img1)

            img1 = image.rotate(img1,torch.bernoulli(0.6)*0.24*(torch.uniform()*2-1),'bilinear')
            im = img1
            pts_anchor, pts_def = warp2d.gen_warp_pts(im:size(), 5, 12)
            im, warpfield = warp2d.warp(im, pts_anchor, pts_def)
            real_A[ind]:copy(im)
        end
    end
--    print('done createrealfake')

    if opt.condition_GAN==1 then
        real_AB = torch.cat(real_A,real_B,2)
    else
        real_AB = real_B -- unconditional GAN, only penalizes structure in B
    end
    
    -- create fake
    local netGoutput = netG:forward(real_A)
    fake_B = netGoutput[1]
    fake_offsetSobel = netGoutput[2]
    fake_offsets = netGoutput[3]
    fake_offsetsBy2 = netGoutput[4]
    fake_offsetsBy4 = netGoutput[5]
    
    if opt.condition_GAN==1 then
        fake_AB = torch.cat(real_A,fake_B,2)
    else
        fake_AB = fake_B -- unconditional GAN, only penalizes structure in B
    end
end

-- create closure to evaluate f(X) and df/dX of discriminator
local fDx = function(x)
    netD:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
    netG:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
    
    gradParametersD:zero()
    
    -- Real
    local output = netD:forward(real_AB)
    local label = torch.FloatTensor(output:size()):fill(real_label)
    if opt.gpu>0 then 
    	label = label:cuda()
    end
    
    local errD_real = criterion:forward(output, label)
    local df_do = criterion:backward(output, label)
    netD:backward(real_AB, df_do)
    
    -- Fake
    local output = netD:forward(fake_AB)
    label:fill(fake_label)
    local errD_fake = criterion:forward(output, label)
    local df_do = criterion:backward(output, label)
    netD:backward(fake_AB, df_do)
    
    errD = (errD_real + errD_fake)/2
    
    return errD, gradParametersD
end

-- create closure to evaluate f(X) and df/dX of generator
local fGx = function(x)
    netD:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
    netG:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
    
    gradParametersG:zero()
    
    -- GAN loss
    local df_dg = torch.zeros(fake_B:size())
    if opt.gpu>0 then 
    	df_dg = df_dg:cuda();
    end
    
    if opt.use_GAN==1 then
       local output = netD.output -- netD:forward{input_A,input_B} was already executed in fDx, so save computation
       local label = torch.FloatTensor(output:size()):fill(real_label) -- fake labels are real for generator cost
       if opt.gpu>0 then 
       	label = label:cuda();
       	end
       errG = criterion:forward(output, label)
       local df_do = criterion:backward(output, label)
       df_dg = netD:updateGradInput(fake_AB, df_do):narrow(2,fake_AB:size(2)-output_nc+1, output_nc)
    else
        errG = 0
    end
    
    -- unary loss
    local df_do_AE = torch.zeros(fake_B:size())
    if opt.gpu>0 then 
    	df_do_AE = df_do_AE:cuda();
    end
    if opt.use_L1==1 then
       errL1 = criterionAE:forward(fake_B, real_B)
       df_do_AE = criterionAE:backward(fake_B, real_B)
    else
        errL1 = 0
    end

    local df_sobel_ = fake_offsetSobel:clone():fill(0)
    local zero_sobel = fake_offsetSobel:clone():fill(0)
    errSobel = criterionSobel:forward(fake_offsetSobel, zero_sobel)
    df_sobel_ = criterionSobel:backward(fake_offsetSobel, zero_sobel) 
    df_sobel_ = df_sobel_:mul(opt.lambda)

    local lossTensor = netLoss:forward({real_A,real_B,fake_offsetsBy2,fake_offsetsBy4})
    local lossTensorBy2 = lossTensor[1]
    local lossTensorBy4 = lossTensor[2]
    local zero_by2 = lossTensorBy2:clone():fill(0)
    local zero_by4 = lossTensorBy4:clone():fill(0)
    local errLossBy2 = criterionAEby2:forward(lossTensorBy2,zero_by2)
    local errLossBy4 = criterionAEby4:forward(lossTensorBy4,zero_by4)
    local df_by2_ = criterionAEby2:backward(lossTensorBy2,zero_by2)
    local df_by4_ = criterionAEby4:backward(lossTensorBy4,zero_by4)
    local df_offsets_ = netLoss:updateGradInput({real_A,real_B,fake_offsetsBy2,fake_offsetsBy4},{df_by2_,df_by4_})
    local df_offsetsBy2_ = df_offsets_[3]:mul(opt.lambda)
    local df_offsetsBy4_ = df_offsets_[4]:mul(opt.lambda)

   
    local df__ = df_dg + df_do_AE:mul(opt.lambda)
    if opt.useSobel == 0 then 
      netG:backward(real_A, {df__, df_sobel_:clone():fill(0),df_sobel_:clone():fill(0), df_offsetsBy2_,df_offsetsBy4_})
    else
      netG:backward(real_A, {df__, df_sobel_, df_sobel_:clone():fill(0), df_offsetsBy2_,df_offsetsBy4_})
    end
    return errG, gradParametersG
end

require 'image';
function Offsets2HSV(offsets)
    local nDim = offsets:size():size()
    local xOffsets = offsets:select(nDim-2,2)
    local yOffsets = offsets:select(nDim-2,1)
    local mag = torch.sqrt(torch.pow(xOffsets,2)+torch.pow(yOffsets,2))
    mag = mag/mag:max()
    local ang = torch.atan2(yOffsets,xOffsets)
    ang = (ang*180)/math.pi
    ang = ang/ang:max()
    local shape = offsets:size()
    shape[nDim-2] = 3
    local hsv = torch.Tensor(shape):typeAs(offsets):fill(1)
    hsv:select(nDim-2,1)[{}] = ang
    hsv:select(nDim-2,3)[{}] = mag
    local hsvrgb = image.hsv2rgb(hsv)
    return hsvrgb
end

-- train
local best_err = nil
paths.mkdir(opt.checkpoints_dir)
paths.mkdir(opt.checkpoints_dir .. '/' .. opt.name)

-- save opt
file = torch.DiskFile(paths.concat(opt.checkpoints_dir, opt.name, 'opt.txt'), 'w')
file:writeObject(opt)
file:close()

-- parse diplay_plot string into table
opt.display_plot = string.split(string.gsub(opt.display_plot, "%s+", ""), ",")
for k, v in ipairs(opt.display_plot) do
    if not util.containsValue({"errG", "errD", "errL1"}, v) then 
        error(string.format('bad display_plot value "%s"', v)) 
    end
end

-- display plot config
local plot_config = {
  title = "Loss over time",
  labels = {"epoch", unpack(opt.display_plot)},
  ylabel = "loss",
}

-- display plot vars
local plot_data = {}
local plot_win

local counter = 0
for epoch = 1, opt.niter do
    epoch_tm:reset()
    for i = 1, math.min(data:size(), opt.ntrain), opt.batchSize do
        tm:reset()
        
        -- load a batch and run G on that batch
        createRealFake()
        
        -- (1) Update D network: maximize log(D(x,y)) + log(1 - D(x,G(x)))
        if opt.use_GAN==1 then optim.adam(fDx, parametersD, optimStateD) end
        
        -- (2) Update G network: maximize log(D(x,G(x))) + L1(y,G(x))
        optim.adam(fGx, parametersG, optimStateG)

        -- display
        counter = counter + 1
        if counter % opt.display_freq == 0 and opt.display then
            createRealFake()
            if opt.preprocess == 'colorization' then 
                local real_A_s = util.scaleBatch(real_A:float(),100,100)
                local fake_B_s = util.scaleBatch(fake_B:float(),100,100)
                local real_B_s = util.scaleBatch(real_B:float(),100,100)
                disp.image(util.deprocessL_batch(real_A_s), {win=opt.display_id, title=opt.name .. ' input'})
                disp.image(util.deprocessLAB_batch(real_A_s, fake_B_s), {win=opt.display_id+1, title=opt.name .. ' output'})
                disp.image(util.deprocessLAB_batch(real_A_s, real_B_s), {win=opt.display_id+2, title=opt.name .. ' target'})
            else
                disp.image(util.deprocess_batch(util.scaleBatch(real_A:float(),100,100)), {win=opt.display_id, title=opt.name .. ' input'})
                disp.image(util.deprocess_batch(util.scaleBatch(fake_B:float(),100,100)), {win=opt.display_id+1, title=opt.name .. ' output'})
                disp.image(util.deprocess_batch(util.scaleBatch(real_B:float(),100,100)), {win=opt.display_id+2, title=opt.name .. ' target'})
            end
        end
      
        -- write display visualization to disk
        --  runs on the first batchSize images in the opt.phase set
        if counter % opt.save_display_freq == 0 and opt.display then
            local serial_batches=opt.serial_batches
            opt.serial_batches=1
            opt.serial_batch_iter=1
            
            local image_out = nil
            local N_save_display = 10 
            local N_save_iter = torch.max(torch.Tensor({1, torch.floor(N_save_display/opt.batchSize)}))
            for i3=1, N_save_iter do
            
                createRealFake()
                print('save to the disk')
                if opt.preprocess == 'colorization' then 
                    for i2=1, fake_B:size(1) do
                        if image_out==nil then image_out = torch.cat(util.deprocessL(real_A[i2]:float()),util.deprocessLAB(real_A[i2]:float(), fake_B[i2]:float()),3)/255.0
                        else image_out = torch.cat(image_out, torch.cat(util.deprocessL(real_A[i2]:float()),util.deprocessLAB(real_A[i2]:float(), fake_B[i2]:float()),3)/255.0, 2) end
                    end
                else
                    for i2=1, fake_B:size(1) do
                        if image_out==nil then image_out = torch.cat({util.deprocess(real_A[i2]:float()),util.deprocess(fake_B[i2]:float()),Offsets2HSV(fake_offsets[i2]:float())},3)
                        else image_out = torch.cat(image_out, torch.cat({util.deprocess(real_A[i2]:float()),util.deprocess(fake_B[i2]:float()),Offsets2HSV(fake_offsets[i2]:float())},3), 2) end
                    end
                end
            end
            image.save(paths.concat(opt.checkpoints_dir,  opt.name , counter .. '_train_res.png'), image_out)
            
            opt.serial_batches=serial_batches
        end
        
        -- logging and display plot
        if counter % opt.print_freq == 0 then
            local loss = {errG=errG and errG or -1, errD=errD and errD or -1, errL1=errL1 and errL1 or -1, errSobel=errSobel and errSobel or -1}
            local curItInBatch = ((i-1) / opt.batchSize)
            local totalItInBatch = math.floor(math.min(data:size(), opt.ntrain) / opt.batchSize)
            print(('Epoch: [%d][%8d / %8d]\t Time: %.3f  DataTime: %.3f  '
                    .. '  Err_G: %.4f  Err_D: %.4f  ErrL1: %.4f ErrSobel: %.4f'):format(
                     epoch, curItInBatch, totalItInBatch,
                     tm:time().real / opt.batchSize, data_tm:time().real / opt.batchSize,
                     errG, errD, errL1, errSobel))
           
            local plot_vals = { epoch + curItInBatch / totalItInBatch }
            for k, v in ipairs(opt.display_plot) do
              if loss[v] ~= nil then
               plot_vals[#plot_vals + 1] = loss[v] 
             end
            end

            -- update display plot
            if opt.display then
              table.insert(plot_data, plot_vals)
              plot_config.win = plot_win
              plot_win = disp.plot(plot_data, plot_config)
            end
        end
        
        -- save latest model
        if counter % opt.save_latest_freq == 0 then
            print(('saving the latest model (epoch %d, iters %d)'):format(epoch, counter))
            torch.save(paths.concat(opt.checkpoints_dir, opt.name, 'latest_net_G.t7'), netG:clearState())
            torch.save(paths.concat(opt.checkpoints_dir, opt.name, 'latest_net_D.t7'), netD:clearState())
        end
        
    end
    
    
    parametersD, gradParametersD = nil, nil -- nil them to avoid spiking memory
    parametersG, gradParametersG = nil, nil
    
    if epoch % opt.save_epoch_freq == 0 then
        torch.save(paths.concat(opt.checkpoints_dir, opt.name,  epoch .. '_net_G.t7'), netG:clearState())
        torch.save(paths.concat(opt.checkpoints_dir, opt.name, epoch .. '_net_D.t7'), netD:clearState())
    end
    
    print(('End of epoch %d / %d \t Time Taken: %.3f'):format(
            epoch, opt.niter, epoch_tm:time().real))
    parametersD, gradParametersD = netD:getParameters() -- reflatten the params and get them
    parametersG, gradParametersG = netG:getParameters()
end
