function results=run_FCN2(seq, res_path, bSaveImage)

addpath('utilities/');
addpath('/home/vis/zhangzhe/caffe-future/matlab/caffe');
caffePath = '/home/vis/zhangzhe/caffe-future/';
input_size = [128,128];
duration = 0;
rects = zeros(seq.len,4);
for imgNum = 1:seq.len
    tic;
    imgName = seq.s_frames{imgNum};
    imgIn = imread(imgName);
    if ndims(imgIn)<3
        imgIn = repmat(imgIn, [1,1,3]);
    end
    if imgNum==1
        rect = seq.init_rect;
        modelFile = [caffePath 'fcn-tracker/fcn-8s-pascal-deploy2.prototxt'];
        pretrainedFile = [caffePath 'fcn-tracker/fcn-8s-pascal.caffemodel'];
        caffe('set_device', 0);
        caffe('init', modelFile, pretrainedFile, 'test');
        caffe('set_mode_gpu');
        caffe('presolve');
    end
    [h,w,k] = size(imgIn);
    target_sz = max(input_size(1), 4*sqrt(rect(3)*rect(4)));
    ratio = input_size/target_sz;
    xs = round(rect(1)+rect(3)/2-target_sz/2: rect(1)+rect(3)/2+target_sz/2-1);
    ys = round(rect(2)+rect(4)/2-target_sz/2: rect(2)+rect(4)/2+target_sz/2-1);
    xs(xs<1) = 1;
    ys(ys<1) = 1;
    xs(xs>w) = w;
    ys(ys>h) = h;
    img = imgIn(ys,xs,:);
    img = imresize(img, input_size);
    img = double(img(:,:,[3,2,1]));
    img(:,:,1) = img(:,:,1)-104.00698793; 
    img(:,:,2) = img(:,:,2)-116.66876762; 
    img(:,:,3) = img(:,:,3)-122.67891434;
    output = caffe('forward', {single(img);single(zeros(input_size))}, 'data_input_0_split', 'conv9-1', 'conv9-1');
    output = output{1};
    output = squeeze(output);
    cos_win = hann(input_size(1))*hann(input_size(2))';
    output = output.*cos_win;
    filter = fspecial('gaussian', [5,5], 1);
    output = imfilter(output,filter);

    if imgNum > 1
        [x, y] = find(output == max(output(:)), 1);
        dx = (x-floor(input_size(2)/2))/ratio(2);
        dy = (y-floor(input_size(1)/2))/ratio(1);
        rect = rect+fix([dy,dx,0,0]);
        rect(1) = max(rect(1),1);
        rect(2) = max(rect(2),1);
        rect(1) = min(rect(1),w-rect(3));
        rect(2) = min(rect(2),h-rect(4));
    end
    sigma = 4;
    if imgNum==1
        x = floor(input_size(2)/2);
        y = floor(input_size(1)/2);
    end
    [xx,yy] = ndgrid((1:input_size(2)) - x, (1:input_size(1)) - y);
    label = exp(-0.5 / sigma^2 * (xx.^2 + yy.^2));

    if imgNum == 1
        iterNum = 100;
        start_layer =  'fc6';
        for i = 1:iterNum
            lr = 1e-10;
            output = caffe('forward', {single(img);single(label)}, start_layer, 'loss', 'loss');
            output = output{1};
            caffe('backward', {single(output)}, 'loss', start_layer);
            caffe('update', lr);
        end
    else
        iterNum = 1;
        start_layer = 'fc6';
        for i = 1:iterNum
            lr = 1e-11;
            output = caffe('forward', {single(img);single(label)}, 'conv9-1', 'loss', 'loss');
            output = output{1};
            if output > 1000
                break;
            end
            caffe('backward', {single(output)}, 'loss', start_layer);
            caffe('update', lr);
        end
    end
    rects(imgNum,:) = rect;
    if imgNum>1
        duration = duration + toc;
    end
end
close('all');
fprintf('%d frames took %.3f seconds : %.3ffps\n',seq.len-1,duration,(seq.len-1)/duration);
results.type = 'rect';
results.res = rects;
results.fps =  (seq.len-1)/duration;
end
