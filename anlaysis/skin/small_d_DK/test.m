im_path1 = '\\fatherserverdw\andre\Alzherimers brain\Fully Sectioned\FXFAD12FB2\1x_python\testing\FXFAD12FB2_0035.tif';
im_path2 = '\\fatherserverdw\andre\Alzherimers brain\Fully Sectioned\FXFAD12FB2\1x_python\testing\FXFAD12FB2_0037.tif';

im1 = imread(im_path1);
im2 = imread(im_path2);

size_im1 = size(im1);
size_im2 = size(im2);

max_x = max(size_im1(1), size_im2(1));
max_y = max(size_im1(2), size_im2(2));

im1_resized = imresize(im1, [max_x, max_y]);
im2_resized = imresize(im2, [max_x, max_y]);

overlapImage = im1_resized & im2_resized;

imshow(im1_resized);
imshow(im2_resized);

imshow(overlapImage)

numOverlapPixels = nnz(overlapImage);
%%
pth_1x = '\\fatherserverdw\andre\Alzherimers brain\Fully Sectioned\FXFAD12FB2\1x_python\';

imlist=dir([pth_1x,'*tif']);
%%

max_size=[0 0];
for kk=1:length(imlist)
    image_info=imfinfo([pth_1x,imlist(kk).name]);
    max_size=[max([max_size(1),image_info.Height]) max([max_size(2),image_info.Width])];
end

%%




















