function [flip_x, flip_y, max_angle] = find_best_angle(amv, arf)

    % % make reference image tissue area image
    % ref_image_name = imlist(i).name;
    % ref_image=imread([pth_1x,ref_image_name]);
    % ref_image_resized = imresize(ref_image, [max_size(1) max_size(2)]);
    % ref_TA=find_tissue_area(ref_image_resized);
    % 
    % % make moving image tissue area image
    % mov_image_name = imlist(i+1).name;
    % mov_image=imread([pth_1x,mov_image_name]);
    % mov_image_resized = imresize(mov_image, [max_size(1) max_size(2)]);
    % mov_TA=find_tissue_area(mov_image_resized);

    ref_TA = arf;
    mov_TA= amv;

    ref_centroid = get_centroid(arf);
    mov_centroid = get_centroid(amv);

    [padded_ref_TA, padded_mov_TA] = align_centroids(ref_TA, mov_TA, ref_centroid, mov_centroid);

    % pad a lil more
    pad_amt=200;  % helps align centroids closer to each other later on due to rounding error
    padded_ref_TA = padarray(padded_ref_TA, [pad_amt, pad_amt], 0, 'both');
    padded_mov_TA = padarray(padded_mov_TA, [pad_amt, pad_amt], 0, 'both');

    ref_centroid = get_centroid(padded_ref_TA);
    mov_centroid = get_centroid(padded_mov_TA);

    degrees = 45/16;



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
    original_overlaps = test_angles(padded_mov_TA, padded_ref_TA, mov_centroid, degrees, original_overlaps);
    flip_x_overlaps = test_angles(flipped_x_padded_mov_TA, flipped_x_padded_ref_TA, flipped_x_mov_centroids, degrees, flip_x_overlaps);
    flip_y_overlaps = test_angles(flipped_y_padded_mov_TA, flipped_y_padded_ref_TA, flipped_y_mov_centroids, degrees, flip_y_overlaps);
    % flip_x_y_overlaps = test_angles(flipped_x_y_padded_mov_TA, flipped_x_y_padded_ref_TA, flipped_x_y_mov_centroids, 45, flip_x_y_overlaps);

    
    max_og_angle_val = max(original_overlaps);
    max_flip_y_angle_val = max(flip_y_overlaps);
    max_flip_x_angle_val = max(flip_x_overlaps);

    flip_x = false;
    flip_y = false;

    arbitary_value = 1.05;  % so that flipping x and y is harder than keeping original

    if max_flip_x_angle_val >= max_og_angle_val*arbitary_value
        disp('flip x')
        flip_x = true;
        max_angle = find(flip_x_overlaps == max_flip_x_angle_val)*degrees;

    elseif max_flip_y_angle_val >= max_og_angle_val*arbitary_value
        flip_y = true;
        disp('flip y')
        max_angle = find(flip_y_overlaps == max_flip_y_angle_val)*degrees;

    else
        disp('no flip')
        max_angle = find(original_overlaps == max_og_angle_val)*degrees;
    end

    if ~isscalar(max_angle)
        max_angle = max_angle(1);
    end

    % if max_angle == 0
    %     max_angle = 360 - degrees;
    % else
    %     max_angle = max_angle - degrees; % to get back to correct value
    % end

    disp('max angle: ');disp(max_angle);


end