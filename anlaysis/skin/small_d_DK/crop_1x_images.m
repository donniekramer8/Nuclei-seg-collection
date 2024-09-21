pth_1x = '\\fatherserverdw\andre\Alzherimers brain\Fully Sectioned\FXFAD12FB2\1x_python\need_cropped\';
pthout = '\\fatherserverdw\andre\Alzherimers brain\Fully Sectioned\FXFAD12FB2\1x_python\cropped\';
pthout_im_name = '\\fatherserverdw\andre\Alzherimers brain\Fully Sectioned\FXFAD12FB2\1x_python\cropped\';

imlist=dir([pth_1x,'*tif']);

for i=1:length(imlist)
    if ~exist([pthout_im_name, imlist(i).name], "file")
        image=imread([pth_1x, imlist(i).name]);
        image_cropped=imcrop(image);
        imwrite(image_cropped, [pthout_im_name, imlist(i).name])
    end
end