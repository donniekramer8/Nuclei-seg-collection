im = imread('\\10.99.68.178\andreex\students\Donald Monkey fetus\data\Skin_Xenium_slide\Xenium_Prime_Human_Skin_FFPE_he_image.ome.tif');
imshow(im)

ds=8;
im_ds=im(1:ds:end,1:ds:end,:);

imshow(im_ds)
aff = [[0.013437495	1.288182534	-1168.501329];
[-1.288182534	0.013437495	21910.70207];
[0	0	1]];

aff = affinetform2d([[0.013437495	1.288182534	-1168.501329];
[-1.288182534	0.013437495	21910.70207];
[0	0	1]]);

reg_im = imwarp(im_ds,aff);

imshow(reg_im)

%%
morph = imread('\\10.99.68.178\andreex\students\Donald Monkey fetus\data\Skin_Xenium_slide\Xenium_Prime_Human_Skin_FFPE_outs\morphology.ome.tif');