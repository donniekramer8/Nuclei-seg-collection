pth = '\\10.162.80.16\Andre_expansion\data\monkey_fetus\Stardist\StarDist_12_25_23\volcell_coords_with_features\';
matlist = dir([pth, '*.mat']);

% This code is going to take each individual features df for each slide of 
% the monkey and make a save a new version with normalized staining
% intensity means instead of raw intensity mean

outpth = '\\10.162.80.16\Andre_expansion\data\monkey_fetus\Stardist\StarDist_12_25_23\volcell_coords_with_features\with_norm_intensities\';


%%
% make empty list to show line plot of staining intensities at end (is cool)
r_means = zeros(1,length(matlist));
g_means = zeros(1,length(matlist));
b_means = zeros(1,length(matlist));

for i=1:length(matlist)
    disp(i)

    outnm = [outpth, matlist(i).name];
    if ~exist("outnm","file")

        load([pth, matlist(i).name], "df", "colnames")
    
        r_col = df(:,end-2);
        g_col = df(:,end-1);
        b_col = df(:,end);
    
        r_mean = mean(r_col);
        r_std = std(r_col);
    
        g_mean = mean(g_col);
        g_std = std(g_col);
    
        b_mean = mean(b_col);
        b_std = std(b_col);
    
        % add means to lists to make cool line plot
        r_means(i) = r_mean;g_means(i) = g_mean;b_means(i) = b_mean;
    
        % make new column with normalized values
        norm_r_int_col = (r_col-r_mean)/r_std;
        norm_g_int_col = (g_col-g_mean)/g_std;
        norm_b_int_col = (b_col-b_mean)/b_std;
    
        % replace old column with normalized cols
        df(:,end-2) = norm_r_int_col;
        df(:,end-1) = norm_g_int_col;
        df(:,end) = norm_b_int_col;
    
        % uncomment probably if using this
        % save([outpth, matlist(i).name], "df", "colnames");
    
        %disp([r_mean, g_mean, b_mean])
        %disp([r_std, g_std, b_std])
    
        % disp('')
    end
end

%%
figure(3);hold on;
plot(1:length(r_means), r_means, 'r')
plot(1:length(g_means), g_means, 'g')
plot(1:length(b_means), b_means, 'b')

% ylim([75, 200]);
ylim([-.001, .001]);
ylabel("Mean Intensity", 'FontSize', 20);
xlabel("Slide Number", 'FontSize', 20);
title("Mean RGB Intensities Inside Nucleus Contours", 'FontSize', 20);
legend("R Channel", "G Channel", "B Channel", 'FontSize', 16);

hold off;

%% make post-norm graph

pth = '\\10.162.80.16\Andre_expansion\data\monkey_fetus\Stardist\StarDist_12_25_23\volcell_coords_with_features\with_norm_intensities\';
matlist = dir([pth, '*.mat']);

% make empty list to show line plot of staining intensities
r_means = zeros(1,length(matlist));
g_means = zeros(1,length(matlist));
b_means = zeros(1,length(matlist));

for i=1:length(matlist)
    disp(i)

    if ~exist("outnm","file")

        load([pth, matlist(i).name], "df", "colnames")
    
        r_col = df(:,end-2);
        g_col = df(:,end-1);
        b_col = df(:,end);
    
        r_mean = mean(r_col);
        g_mean = mean(g_col);
        b_mean = mean(b_col);
    
        % add means to lists to make cool line plot
        r_means(i) = r_mean;g_means(i) = g_mean;b_means(i) = b_mean;
    end
end




