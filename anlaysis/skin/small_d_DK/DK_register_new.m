%% Make data

pth_1x = '\\fatherserverdw\andre\Alzherimers brain\Fully Sectioned\FXFAD12FB2\1x_python\';

imlist=dir([pth_1x,'*tif']);

max_size=[0 0];
for kk=1:length(imlist)
    image_info=imfinfo([pth_1x,imlist(kk).name]);
    max_size=[max([max_size(1),image_info.Height]) max([max_size(2),image_info.Width])];
end

%% Process data

function [flip_x, flip_y, max_angle] = find_best_angle()
for i=1:(length(imlist)-1)
    % make reference image tissue area image
    ref_image_name = imlist(i).name;
    ref_image=imread([pth_1x,ref_image_name]);
    ref_image_resized = imresize(ref_image, [max_size(1) max_size(2)]);
    ref_TA=find_tissue_area(ref_image_resized);

    % make moving image tissue area image
    mov_image_name = imlist(i+1).name;
    mov_image=imread([pth_1x,mov_image_name]);
    mov_image_resized = imresize(mov_image, [max_size(1) max_size(2)]);
    mov_TA=find_tissue_area(mov_image_resized);

    ref_centroid = get_centroid(ref_TA);
    mov_centroid = get_centroid(mov_TA);

    [padded_ref_TA, padded_mov_TA] = align_centroids(ref_TA, mov_TA, ref_centroid, mov_centroid);

    % pad a lil more
    pad_amt=1000;
    padded_ref_TA = padarray(padded_ref_TA, [pad_amt, pad_amt], 0, 'both');
    padded_mov_TA = padarray(padded_mov_TA, [pad_amt, pad_amt], 0, 'both');

    ref_centroid = get_centroid(padded_ref_TA);
    mov_centroid = get_centroid(padded_mov_TA);

    degrees = 45;



    original_overlaps = zeros(1, (360/degrees));
    flip_x_overlaps = zeros(1, (360/degrees));
    flip_y_overlaps = zeros(1, (360/degrees));
    flip_x_y_overlaps = zeros(1, (360/degrees));

    flipped_x_padded_ref_TA = padded_ref_TA;
    flipped_x_padded_mov_TA = fliplr(padded_mov_TA);
    [flipped_x_padded_ref_TA, flipped_x_padded_mov_TA] = align_centroids(flipped_x_padded_ref_TA, flipped_x_padded_mov_TA, get_centroid(flipped_x_padded_ref_TA), get_centroid(flipped_x_padded_mov_TA));
    flipped_x_mov_centroids = get_centroid(flipped_x_padded_mov_TA);

    flipped_y_padded_ref_TA = padded_ref_TA;
    flipped_y_padded_mov_TA = flipud(padded_mov_TA);
    [flipped_y_padded_ref_TA, flipped_y_padded_mov_TA] = align_centroids(flipped_y_padded_ref_TA, flipped_y_padded_mov_TA, get_centroid(flipped_y_padded_ref_TA), get_centroid(flipped_y_padded_mov_TA));
    flipped_y_mov_centroids = get_centroid(flipped_y_padded_mov_TA);

    % flipped_x_y_padded_ref_TA = padded_ref_TA;
    % flipped_x_y_padded_mov_TA = flipud(fliplr(padded_mov_TA));
    % [flipped_x_y_padded_ref_TA, flipped_x_y_padded_mov_TA] = align_centroids(flipped_x_y_padded_ref_TA, flipped_x_y_padded_mov_TA, get_centroid(flipped_x_y_padded_mov_TA), get_centroid(flipped_x_y_padded_mov_TA));
    % flipped_x_y_mov_centroids = get_centroid(flipped_x_y_padded_mov_TA);

    
    % Rotate function
    original_overlaps = test_angles(padded_mov_TA, padded_ref_TA, mov_centroid, 45, original_overlaps);
    flip_x_overlaps = test_angles(flipped_x_padded_mov_TA, flipped_x_padded_ref_TA, flipped_x_mov_centroids, 45, flip_x_overlaps);
    flip_y_overlaps = test_angles(flipped_y_padded_mov_TA, flipped_y_padded_ref_TA, flipped_y_mov_centroids, 45, flip_y_overlaps);
    % flip_x_y_overlaps = test_angles(flipped_x_y_padded_mov_TA, flipped_x_y_padded_ref_TA, flipped_x_y_mov_centroids, 45, flip_x_y_overlaps);

    
    max_og_angle_val = max(original_overlaps);
    max_flip_y_angle_val = max(flip_y_overlaps);
    max_flip_x_angle_val = max(flip_x_overlaps);

    flip_x = false;
    flix_y = false;

    if max_flip_y_angle_val >= max_og_angle_val
        flip_y = true;
        max_angle = find(array == max_flip_y_angle_val)*degrees;
    elseif max_flip_x_angle_val >= max_og_angle_val
        flip_x = true;
        max_angle = find(array == max_flip_x_angle_val)*degrees;
    else
        max_angle = find(array == max_og_angle_val)*degrees;
    end





    % for i=degrees:degrees:360
    %     disp('start');
    %     rotated_mov_TA = rotateAround(padded_mov_TA, mov_centroid, i);
    % 
    %     imshow(rotated_mov_TA);
    %     hold on;
    %     plot(mov_centroid(1), mov_centroid(2), 'r+', 'MarkerSize', 10);
    %     hold off;
    %     disp('done');
    % 
    %     overlapImage = padded_ref_TA & rotated_mov_TA;
    % 
    %     numOverlapPixels = nnz(overlapImage);
    %     disp(numOverlapPixels)
    %     disp('')
    % 
    % end

    disp('')










end

% 
% 
% image_folder = '\\fatherserverdw\andre\Alzherimers brain\Fully Sectioned\FXFAD12FB2\1x_python\';
% image_name='FXFAD12FB2_0037';
% tp='.tif';
% image = imread([image_folder,image_name,tp]);
% 
% pthTA='\\fatherserverdw\andre\Alzherimers brain\Fully Sectioned\FXFAD12FB2\1x_python\testing\';
% 
% TA=find_tissue_area(image);
% imwrite(TA,[pthTA,image_name,tp]);
% 
% image = TA;