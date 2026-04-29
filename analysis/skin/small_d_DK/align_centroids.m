function [ref_TA,mov_TA] = align_centroids(ref_TA, mov_TA, ref_centroid, mov_centroid)
% Pads two images so that their centroids line up

    ref_centroid_x = ref_centroid(1);
    ref_centroid_y = ref_centroid(2);

    mov_centroid_x = mov_centroid(1);
    mov_centroid_y = mov_centroid(2);

    if ref_centroid_x > mov_centroid_x
        displacement_x = round(ref_centroid_x - mov_centroid_x);
        % pad left side of mov_TA by displacement_x
        mov_TA = padarray(mov_TA, [0, displacement_x], 'pre');
        % pad right side of ref_TA by displacement_x
        ref_TA = padarray(ref_TA, [0, displacement_x], 'post');
    else
        displacement_x = round(mov_centroid_x - ref_centroid_x);
        % pad left side of ref_TA by displacement_x
        ref_TA = padarray(ref_TA, [0, displacement_x], 'pre');
        % pad right side of mov_TA by displacement_x
        mov_TA = padarray(mov_TA, [0, displacement_x], 'post');
    end

    if ref_centroid_y > mov_centroid_y
        displacement_y = round(ref_centroid_y - mov_centroid_y);
        % pad top side of mov_TA by displacement_y
        mov_TA = padarray(mov_TA, [displacement_y, 0], 'pre');
        % pad bottom side of ref_TA by displacement_y
        ref_TA = padarray(ref_TA, [displacement_y, 0], 'post');
    else
        displacement_y = round(mov_centroid_y - ref_centroid_y);
        % pad top side of ref_TA by displacement_y
        ref_TA = padarray(ref_TA, [displacement_y, 0], 'pre');
        % pad bottom side of mov_TA by displacement_y
        mov_TA = padarray(mov_TA, [displacement_y, 0], 'post');
    end