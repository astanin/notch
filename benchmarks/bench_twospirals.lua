-- Create a 2-50-50-50-50-10-1 multilayer perceptron with Torch7
-- and train it on the twospirals problem.
--
-- Major to notch C++ implementation (bench_twospirals.cpp):
--
--  * torch version is using Tanh, notch version is using scaledTanh
--  * torch version is usuing fixed rate SGD, notch version is using ADADELTA
--
-- Feel free to suggest how the script can be improved.
--
require 'torch'
require 'nn'

-- create dataset compatible with torch7
function loadDataset(diskfile)
    -- use csv2t7.sh to convert CSV files to torch native data format:
    -- https://github.com/locklin/torch-things/blob/master/csv2t7.sh
    local file = torch.DiskFile(diskfile)
    local table = file:readObject()

    local n_samples = table:size()[1]

    local inputs = table[{{},{1,2}}] -- the first two columns are inputs
    local labels = table[{{},3}]     -- the third column is a label

    -- we follow Torch nn example here:
    -- https://github.com/torch/nn/blob/master/doc/training.md
    local dataset = {}
    function dataset:size() return n_samples end
    for i=1, n_samples do
        local input = torch.Tensor(2)
        local label = torch.Tensor(1)
        input[1] = inputs[i][1]
        input[2] = inputs[i][2]
        label[1] = labels[i]
        dataset[i] = {input, label}
    end
    return dataset
end

trainset = loadDataset('twospirals-train.t7')
testset = loadDataset('twospirals-test.t7')

net = nn.Sequential()
net:add(nn.Linear(2, 50))
net:add(nn.Tanh()) -- Torch7 nn package lacks scaled Tanh activation
net:add(nn.Linear(50, 50))
net:add(nn.Tanh())
net:add(nn.Linear(50, 50))
net:add(nn.Tanh())
net:add(nn.Linear(50, 50))
net:add(nn.Tanh())
net:add(nn.Linear(50, 10))
net:add(nn.Tanh())
net:add(nn.Linear(10, 1))
loss = nn.MSECriterion()

-- I don't find examples how to use optim.adadelta with StochasticGradient
-- in torch; I stick to plain fixed rate SGD in this script;
trainer = nn.StochasticGradient(net, loss)
trainer.learningRate = 1e-3
trainer.maxIteration = 1000
trainer:train(trainset)

