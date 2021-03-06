%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Not sure if this works.                                   %
% All paths are still hardcoded, some of them absolute      %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

path(path,'/export/home2/NoCsBack/hci/wout/caffe/')
path(path,'/export/home2/NoCsBack/hci/wout/caffe/matlab')
path(path,'/export/home2/NoCsBack/hci/wout/caffe')

use_gpu = 1;
gpu_id = 0;

% Set caffe mode
if use_gpu
  caffe.set_mode_gpu();
  caffe.set_device(gpu_id);
else
  caffe.set_mode_cpu();
end

model = 'matlab_features_reference/deploy_features.prototext';
weights = 'data/VGG_ILSVRC_16_layers.caffemodel';
batch_size = 10;

net = caffe.Net(model, weights, 'test');


%% input files spec

root_path = 'Flickr30kEntities/image_snippets/';
fs = textread([root_path 'images.txt'], '%s');
N = length(fs);

%%

% iterate over the images in batches
feats = zeros(4096, N, 'single');
for b=1:batch_size:N

    % enter images, and dont go out of bounds
    Is = {};
    for i = b:min(N,b+batch_size-1)
        disp(fs{i})
        I = imread([root_path fs{i}]);
        if ndims(I) == 2
            I = cat(3, I, I, I); % handle grayscale edge case. Annoying!
        end
        Is{end+1} = I;
    end
    input_data = prepare_images_batch(Is);

    tic;
    scores = net.forward({input_data});
    scores = squeeze(scores{1});
    tt = toc;

    nb = length(Is);
    feats(:, b:b+nb-1) = scores(:,1:nb);
    fprintf('%d/%d = %.2f%% done in %.2fs\n', b, N, 100*(b-1)/N, tt);
end

%% write to file

save([root_path 'Flickr30kEntities/vgg_feats_hdf5.mat'], 'feats', '-v7.3');
save([root_path 'Flickr30kEntities/vgg_feats.mat'], 'feats');
