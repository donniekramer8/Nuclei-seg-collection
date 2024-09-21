function [D,xgg,ygg,xx,yy]=calculate_elastic_registration_copy(imrfR,immvR,TArf,TAmv,sz,bf,di,cutoff)
if ~exist('cutoff','var');cutoff=0.15;end
    cc=10;cc2=cc+1;
    szim=size(immvR);
    m=(sz-1)/2+1;  % offset

    % preprocess reference and moving images
    immvR=double(padarray(immvR,[bf bf],mode(immvR(:))));  % Pad moving image with a border
    immvR=imgaussfilt(immvR,3);  % Apply Gaussian smoothing to the moving image

    imrfR=double(padarray(imrfR,[bf bf],mode(imrfR(:))));  % Pad reference image with a border
    imrfR=imgaussfilt(imrfR,3);  % Apply Gaussian smoothing to the reference image

    TAmv=padarray(TAmv,[bf bf],0);  % Pad moving tissue mask with zeros
    TArf=padarray(TArf,[bf bf],0);  % Pad reference tissue mask with zeros


    % make grid for registration points
    n1=randi(round(di/2),1)+bf+m;  % Starting coordinate (x axis), randomization helps generate variation in starting positions
    n2=randi(round(di/2),1)+bf+m;  % Starting coordinate (y axis)
    [x,y]=meshgrid(n1:di:size(immvR,2)-m-bf,n2:di:size(immvR,1)-m-bf);  % grid of coordinates based on n1 and n2
    x=x(:);y=y(:);  % Flatten matrices into column vectors

    
    % get percentage of tissue in each registration tile
    checkS=zeros(size(x));
    numb=200;
    for b=1:numb:size(x)
        b2=min([b+numb-1 length(x)]);
        ii=getImLocalWindowInd_rf([x(b:b2) y(b:b2)],size(TAmv),m-1,1);
        
        % Extract local windows from the tissue masks
        imcheck=reshape(permute(TAmv(ii),[2 1]),[sz sz size(ii,1)]);
        imcheck2=zeros(size(imcheck));
        imcheck2(cc2:end-cc,cc2:end-cc,:)=imcheck(cc2:end-cc,cc2:end-cc,:);
        mvS=squeeze(sum(sum(imcheck2,1),2)); % indices of image tiles with tissue in them,   % Count tissue pixels in each tile
        
        imcheck=reshape(permute(TArf(ii),[2 1]),[sz sz size(ii,1)]);
        rfS=squeeze(sum(sum(imcheck,1),2)); % indices of image tiles with tissue in them,   % Count tissue pixels in each tile
        
        checkS(b:b2)=min([mvS rfS],[],2);  % Take the minimum count as the percentage of tissue in each tile
    end
    clearvars ii imcheck imcheck2
    checkS=checkS/(sz^2);  % Convert counts to percentages
    
    yg=(y-min(y))/di+1;
    xg=(x-min(x))/di+1;
    xgg0=ones([length(unique(y)) length(unique(x))])*-5000;
    ygg0=ones([length(unique(y)) length(unique(x))])*-5000;
    

    % This loop iterates over the registration points that have a tissue percentage 
    % above the specified cutoff. For each selected tile, it extracts the corresponding 
    % region from the moving and reference images. Then, it calls the reg_ims_ELS 
    % function to perform elastic registration and obtain the X and Y components of 
    % the displacement.

    for kk=find(checkS>cutoff)'
        % setup small tiles
        ii=getImLocalWindowInd_rf([x(kk) y(kk)],size(TAmv),m-1,1);ii(ii==-1)=1;
        immvS=immvR(ii);imrfS=imrfR(ii);
        immvS=reshape(permute(immvS,[2 1]),[sz sz]);
        imrfS=reshape(permute(imrfS,[2 1]),[sz sz]);
        
        % calculate registration for tiles kk
        [X,Y]=reg_ims_ELS(immvS,imrfS,2,1);  % Perform elastic registration on the selected tile

        xgg0(yg(kk),xg(kk))=X;  % Store the X-component of the displacement for each registration point
        ygg0(yg(kk),xg(kk))=Y;  % Store the Y-component of the displacement for each registration point
    end

    % smooth registration grid and make interpolated displacement map
    if max(szim)>2000;szimout=round(szim/5);x=x/5;y=y/5;bf=bf/5;else;szimout=szim;end
    [D,xgg,ygg,xx,yy]=make_final_grids(xgg0,ygg0,bf,x,y,szimout);
    
end
