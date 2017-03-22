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
   use_GAN = 1,                       -- set to 0 to turn off GAN term
   use_L1 = 1,                        -- set to 0 to turn off L1 term
   use_Sobel = 0,                     -- set to 1 to turn on Sobel term
   use_NGC = 0,                     -- set to 1 to turn on Sobel term
   which_model_netD = 'basic', -- selects model to use for netD
   which_model_netG = 'unet',  -- selects model to use for netG
   n_layers_D = 0,             -- only used if which_model_netD=='n_layers'
   lambda = 100,               -- weight on L1 term in objective
   lambdaSobel = 100,          -- weight on Sobel term in objective
   lambdaNGC = 100,            -- weight on NGC term in objective
   no_of_outputs = 1,
   val = 0,
}

-- one-line argument parser. parses enviroment variables to override the defaults
for k,v in pairs(opt)
  do 
    local optoption = tonumber(os.getenv(k)) or os.getenv(k) or opt[k]
    if optoption == "" then
      opt[k] = opt[k]
    else
      opt[k] = optoption
    --opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] 
  end
end
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
local smooth_shadow_label = -1
local NGC_label = 0
 
function defineG(input_nc, output_nc, ngf)
    local netG = nil
    local shadowMap = nil
    if     opt.which_model_netG == "encoder_decoder" then netG = defineG_encoder_decoder(input_nc, output_nc, ngf)
    elseif opt.which_model_netG == "unet" then netG = defineG_unet(input_nc, output_nc, ngf)
    elseif opt.which_model_netG == "unet_128" then netG = defineG_unet_128(input_nc, output_nc, ngf)
    elseif opt.which_model_netG == "unet_exposure" then netG = defineG_unet_exposure(input_nc, output_nc, ngf)
    elseif opt.which_model_netG == "unet_exposure_shadow_map" then netG = defineG_unet_exposure_shadow_map(input_nc, output_nc, ngf)
    elseif opt.which_model_netG == "unet_exposure_simple" then netG = defineG_unet_exposure_simple(input_nc, output_nc, ngf)
    else error("unsupported netG model")
    end
   
    netG:apply(weights_init)

    return netG
end

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
local criterionNGC = nn.AbsCriterion()
local criterionNGC_dw = nn.AbsCriterion()
local criterionSobel_dw = nn.AbsCriterion()
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
local real_A = torch.Tensor(opt.batchSize, input_nc, opt.fineSize, opt.fineSize)
local real_B = torch.Tensor(opt.batchSize, output_nc, opt.fineSize, opt.fineSize)
local fake_B = torch.Tensor(opt.batchSize, output_nc, opt.fineSize, opt.fineSize)
local fake_B_clipped = torch.Tensor(opt.batchSize, output_nc, opt.fineSize, opt.fineSize)
local fake_shadowMapNGC = torch.Tensor(opt.batchSize, output_nc, opt.fineSize, opt.fineSize)
local fake_shadowMapNGC_dw = torch.Tensor(opt.batchSize, output_nc, opt.fineSize, opt.fineSize)
local fake_shadowMap = torch.Tensor(opt.batchSize, output_nc, opt.fineSize, opt.fineSize)
local fake_shadowSobel = torch.Tensor(opt.batchSize, output_nc, opt.fineSize, opt.fineSize)
local fake_shadowSobel_dw = torch.Tensor(opt.batchSize, output_nc, opt.fineSize, opt.fineSize)
local real_AB = torch.Tensor(opt.batchSize, output_nc + input_nc*opt.condition_GAN, opt.fineSize, opt.fineSize)
local fake_AB = torch.Tensor(opt.batchSize, output_nc + input_nc*opt.condition_GAN, opt.fineSize, opt.fineSize)
local errD, errG, errL1, errNGC, errNGC_dw, errSobel, errSobel_dw = 0, 0, 0, 0, 0, 0, 0
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
   if opt.cudnn==1 then
      netG = util.cudnn(netG); netD = util.cudnn(netD);
   end
   netD:cuda(); netG:cuda(); criterion:cuda(); criterionAE:cuda(); criterionSobel:cuda(); criterionNGC:cuda(); criterionNGC_dw:cuda(); criterionSobel_dw:cuda();
   print('done')
else
	print('running model on CPU')
end


local parametersD, gradParametersD = netD:getParameters()
local parametersG, gradParametersG = netG:getParameters()



if opt.display then disp = require 'display' end


function createRealFake()
    -- load real
    data_tm:reset(); data_tm:resume()
    local real_data, data_path = data:getBatch()
    data_tm:stop()
    
    real_A:copy(real_data[{ {}, idx_A, {}, {} }])
    real_B:copy(real_data[{ {}, idx_B, {}, {} }])
    
    if opt.condition_GAN==1 then
        real_AB = torch.cat(real_A,real_B,2)
    else
        real_AB = real_B -- unconditional GAN, only penalizes structure in B
    end
    
    -- create fake
    if opt.which_model_netG == "unet_exposure_shadow_map" then
        local netGoutput = netG:forward(real_A)
        fake_B = netGoutput[1]
        fake_shadowMap = netGoutput[2]
        fake_shadowMapNGC = netGoutput[3]
	fake_shadowMapNGC_dw = netGoutput[4]
        fake_shadowSobel = netGoutput[5]
	fake_shadowSobel_dw = netGoutput[6]
    elseif opt.which_model_netG == "unet_exposure_simple" then
        local netGoutput = netG:forward(real_A)
        fake_B = netGoutput[1]
        fake_B_clipped = netGoutput[2]
        fake_shadowMap = netGoutput[3]
    else
        fake_B = netG:forward(real_A)
    end
    
    if opt.condition_GAN==1 then
        fake_AB = torch.cat(real_A,fake_B,2)
    else
        fake_AB = fake_B -- unconditional GAN, only penalizes structure in B
    end
    local predict_real = netD:forward(real_AB)
    local predict_fake = netD:forward(fake_AB)
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
    
    if opt.which_model_netG == "unet_exposure_shadow_map" then
      local df_sobel_ = fake_shadowSobel:clone():fill(0)
      local df_sobel_dw_ = fake_shadowSobel_dw:clone():fill(0)
      if opt.use_Sobel==1 then
        local zero_sobel = fake_shadowSobel:clone():fill(smooth_shadow_label)
        errSobel = criterionSobel:forward(fake_shadowSobel, zero_sobel)
        df_sobel_ = criterionSobel:backward(fake_shadowSobel, zero_sobel) 
        df_sobel_ = df_sobel_:mul(opt.lambdaSobel)

	local zero_sobel_dw = fake_shadowSobel_dw:clone():fill(smooth_shadow_label)
        errSobel_dw = criterionSobel_dw:forward(fake_shadowSobel_dw, zero_sobel_dw)
        df_sobel_dw_ = criterionSobel_dw:backward(fake_shadowSobel_dw, zero_sobel_dw)
        df_sobel_dw_ = df_sobel_dw_:mul(opt.lambdaSobel)
      end

      local df_NGC_ = fake_shadowMapNGC:clone():fill(0)
      local df_NGC_dw_ = fake_shadowMapNGC_dw:clone():fill(0)
      if opt.use_NGC==1 then
        local zero_NGC = fake_shadowMapNGC:clone():fill(NGC_label)
        errNGC = criterionNGC:forward(fake_shadowMapNGC, zero_NGC)
        df_NGC_ = criterionNGC:backward(fake_shadowMapNGC, zero_NGC) 
        df_NGC_ = df_NGC_:mul(opt.lambdaNGC)

	local zero_NGC_dw = fake_shadowMapNGC_dw:clone():fill(NGC_label)
        errNGC_dw = criterionNGC_dw:forward(fake_shadowMapNGC_dw, zero_NGC_dw)
        df_NGC_dw_ = criterionNGC_dw:backward(fake_shadowMapNGC_dw, zero_NGC_dw)
        df_NGC_dw_ = df_NGC_dw_:mul(opt.lambdaNGC)
      end

      local df__ = df_dg + df_do_AE:mul(opt.lambda)

      netG:backward(real_A, {df__, df__:clone():fill(0), df_NGC_, df_NGC_dw_, df_sobel_, df_sobel_dw_})

    elseif opt.which_model_netG == "unet_exposure_simple" then
      local df__ = df_dg + df_do_AE:mul(opt.lambda)
      netG:backward(real_A, {df__, df__:clone():fill(0), df__:clone():fill(0)})
    else
      netG:backward(real_A, df_dg + df_do_AE:mul(opt.lambda))
    end

    return errG, gradParametersG
end




-- train
local best_err = nil
paths.mkdir(opt.checkpoints_dir)
paths.mkdir(opt.checkpoints_dir .. '/' .. opt.name)

-- save opt
file = torch.DiskFile(paths.concat(opt.checkpoints_dir, opt.name, 'opt.txt'), 'w')
file:writeObject(opt)
file:close()

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
                elseif opt.which_model_netG == "unet_exposure_shadow_map" then
                    for i2=1, fake_B:size(1) do
                        if image_out==nil then image_out = torch.cat({util.deprocess(real_A[i2]:float()),util.deprocess(fake_B[i2]:float()),util.deprocess(fake_shadowMap[i2]:float()),util.deprocess(fake_shadowMapNGC[i2]:float()),util.deprocess(fake_shadowSobel[i2]:float())},3)
                        else image_out = torch.cat(image_out, torch.cat({util.deprocess(real_A[i2]:float()),util.deprocess(fake_B[i2]:float()),util.deprocess(fake_shadowMap[i2]:float()),util.deprocess(fake_shadowMapNGC[i2]:float()),util.deprocess(fake_shadowSobel[i2]:float())},3), 2) end
                    end
                elseif opt.which_model_netG == "unet_exposure_simple" then
                    for i2=1, fake_B:size(1) do
                        if image_out==nil then image_out = torch.cat({util.deprocess(real_A[i2]:float()),util.deprocess(fake_B[i2]:float()),util.deprocess(fake_B_clipped[i2]:float()),util.deprocess(fake_shadowMap[i2]:float())},3)
                        else image_out = torch.cat(image_out, torch.cat({util.deprocess(real_A[i2]:float()),util.deprocess(fake_B[i2]:float()),util.deprocess(fake_B_clipped[i2]:float()),util.deprocess(fake_shadowMap[i2]:float())},3), 2) end
                    end
                else 
                    for i2=1, fake_B:size(1) do
                        if image_out==nil then image_out = torch.cat(util.deprocess(real_A[i2]:float()),util.deprocess(fake_B[i2]:float()),3)
                        else image_out = torch.cat(image_out, torch.cat(util.deprocess(real_A[i2]:float()),util.deprocess(fake_B[i2]:float()),3), 2) end
                    end
                end
            end
            image.save(paths.concat(opt.checkpoints_dir,  opt.name , counter .. '_train_res.png'), image_out)
            
            opt.serial_batches=serial_batches
        end
        
        -- logging
        if counter % opt.print_freq == 0 then
            print(('Epoch: [%d][%8d / %8d]\t Time: %.3f  DataTime: %.3f  '
                    .. '  Err_G: %.4f  Err_D: %.4f  ErrL1: %.4f ErrNGC: %.4f ErrSobel: %.4f'):format(
                     epoch, ((i-1) / opt.batchSize),
                     math.floor(math.min(data:size(), opt.ntrain) / opt.batchSize),
                     tm:time().real / opt.batchSize, data_tm:time().real / opt.batchSize,
                     errG and errG or -1, errD and errD or -1, errL1 and errL1 or -1, errNGC and errNGC or -1, errNGC_dw and errNGC_dw or -1, errSobel and errSobel or -1, errSobel_dw and errSobel_dw or -1))
        end
        
        -- save latest model
        if counter % opt.save_latest_freq == 0 then
            print(('saving the latest model (epoch %d, iters %d)'):format(epoch, counter))
            torch.save(paths.concat(opt.checkpoints_dir, opt.name, 'latest_net_G.t7'), netG:clearState())
            torch.save(paths.concat(opt.checkpoints_dir, opt.name, 'latest_net_D.t7'), netD:clearState())
	    if opt.val == 1 then
	    do
              local command1 = 'DATA_ROOT='..opt.DATA_ROOT..' name='..opt.name..' which_direction='..opt.which_direction..' loadSize='..opt.loadSize..' fineSize='..opt.loadSize..' no_of_outputs='..opt.no_of_outputs..' phase=val th test.lua'
	      local handle1 = io.popen(command1)
	      handle1:close()
	      local command2 = 'mv ./results/'..opt.name..' ./results/'..opt.name..'_'..counter
	      local handle2 = io.popen(command2)
	      handle2:close()
	      print('validation ran ...')
	    end
	    end
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
