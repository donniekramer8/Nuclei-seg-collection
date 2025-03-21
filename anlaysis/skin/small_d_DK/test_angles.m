function array = test_angles(mov_TA, ref_TA, mov_centroid, degrees, array)
% rotates the mov_TA image around its centroid to find its highest overlap
% with the ref_TA, fills up an array with the values and returns that array
    % disp('start');

    % ref_centroid = get_centroid(ref_TA);

    for i=degrees:degrees:360

        rotated_mov_TA = rotateAround(mov_TA, mov_centroid, i);
        % rotated_mov_TA = rotateAround(mov_TA, ref_centroid, i);
        
        % imshowpair(rotated_mov_TA, ref_TA);
        % hold on;
        % plot(mov_centroid(1), mov_centroid(2), 'r+', 'MarkerSize', 10);
        % hold off;

        overlapImage = ref_TA & rotated_mov_TA;

        % figure(13),imshowpair(rotated_mov_TA,ref_TA)

        numOverlapPixels = nnz(overlapImage);

        array(i/degrees) = numOverlapPixels;
    end
    % disp('done');
    % maxi = max(array);
    % max_angle = find(array == maxi)*degrees;
    % rotated_mov_TA = rotateAround(mov_TA, mov_centroid, max_angle);
    % overlapImage = ref_TA & rotated_mov_TA;
    % figure(13),imshowpair(rotated_mov_TA,ref_TA)


end

