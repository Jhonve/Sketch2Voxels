require 'paths'
threads = require 'threads'
require 'util'
local image = require 'image'
local mat = require('matio')

threads.Threads.serialization('threads.sharedserialize')

data = {}
data.__index = data

function data.new(opt)
  local self = {}
  setmetatable(self, data)
  self.opt = opt

  -- initialize variables

  self.depth_train_path = './depth_img/train'
  self.depth_val_path = './depth_img/test'
  self.normal_train_path = './normal_img/train'
  self.normal_val_path = './normal_img/test'
  self.voxel_train_path = './voxels/train'
  self.voxel_val_path = './voxels/test'

  self.all_models_tensor = self.opt.cache_dir .. 'all_voxels_tensor_' .. '.t7'

  
  cats = paths.dir(self.depth_train_path)
  self._size = 0  -- 目录中的文件模型数据数目
  for i, v in ipairs(cats) do   -- 进入某个类型的文件夹类似./data/chair，这边部分test and train
    if v ~= '.' and v ~= '..' then
      cat = cats[i]
      print(v)
      self._size = self._size + 1
    end
  end
  print('TOTAL SIZE ' .. self._size)

  -- NOTE: this is going to take up > 20GB RAM. Will not work on most machines. 
  -- if your RAM isn't sufficient you will have to read directly from disk
  if paths.filep(self.all_models_tensor) then
    all_models = torch.load(self.all_models_tensor)
  else
    all_models = torch.FloatTensor(self._size, 7, 256, 256)  -- 生成一个size*32*32*32/64的张量
    local curindex = 1
    for i, v in ipairs(cats) do 
      if v ~= '.' and v ~= '..' then
        cat = cats[i]
        -- load depth
        cat_img_path = paths.concat(self.depth_train_path, cat)
        print(i .. ', ' .. 'loading ' .. cat_img_path)
        img = image.load(cat_img_path)
        all_models[{curindex, {1, 3}, {}, {}}] = img

        -- load normal
        cat_img_path = paths.concat(self.normal_train_path, cat)
        print(i .. ', ' .. 'loading ' .. cat_img_path)
        img = image.load(cat_img_path)
        all_models[{curindex, {4, 6}, {}, {}}] = img

        -- load mat
        cat_mat_path = paths.concat(self.voxel_train_path, cat:split('.')[1] .. '.mat')
        print(i .. ', ' .. 'loading ' .. cat_img_path)
        mat_voxel = mat.load(cat_mat_path, 'voxels')
        mat_voxel = mat_voxel:reshape(256, 128)
        all_models[{curindex, {7}, {}, {1, 128}}] = mat_voxel
        curindex = curindex + 1
      end
    end
    torch.save(self.all_models_tensor, all_models)
  end
  self.all_models = all_models
  self.dindices = torch.LongTensor(self.all_models:size(1))
  self.dcur = 1
  self:resetAndShuffle()
  return self
end

function shuffle_indices(t)
  for n = t:size(1), 1, -1 do
      local k = math.random(n)
      t[n], t[k] = t[k], t[n]
  end
  return t
end

function data:resetAndShuffle()
  num_models = self.all_models:size(1)
  assert(num_models == self._size)
  for i = 1,num_models do
    self.dindices[i] = i
  end
  shuffle_indices(self.dindices)
  self.dcur = 1
end

function data:getBatch(quantity)
  local minindex = self.dcur
  local maxindex = math.min(self.all_models:size(1), minindex + quantity - 1)

  local data = self.all_models:index(1, self.dindices[{{minindex,maxindex}}])
  if self.opt.gpu > 0 then
    data = data:cuda()
  end
  collectgarbage()
  collectgarbage()
  self.dcur = self.dcur + (maxindex - minindex + 1)
  cor_batchSize = maxindex - minindex + 1
  return data[{{}, {1, 6}, {}, {}}], data[{{}, 7, {}, {1, 128}}]:reshape(cor_batchSize, 32, 32, 32)
end

function data:size()
  return self._size
end

return data
