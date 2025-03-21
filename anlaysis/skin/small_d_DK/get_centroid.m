function centroid = get_centroid(image)
% Find the indices of pixels with value 1
[row, col] = find(image == 1);

% Calculate the centroid using the mean of the pixel coordinates
centroid_x = mean(col);
centroid_y = mean(row);


% Rin=imref2d(size(image));
% mx=mean(Rin.XWorldLimits);
% my=mean(Rin.YWorldLimits);
% centroid=[mx my];


% Create a plot to visualize the image and centroid
% imshow(image);
% hold on;
% plot(centroid_x, centroid_y, 'r+', 'MarkerSize', 10);
% hold off;

% Return the centroid coordinates
centroid = [centroid_x, centroid_y];
end