% Blind-SIM reconstruction
clear all
close all
%% inputs
frames = 3;
upsample = 'True';

svpath = 'Z:/Users/ew535/20220427_Chiara/to process/488_561_640_1312n1_SHSY5Y_8.tif';

data = double(getNormTifStack(svpath));
data = data./max(data(:));
[X,Y,t] = size(data);
sections = floor(t/frames);
output = zeros([X Y sections]);

svpath = svpath(1:end-4);
svpath = strcat(svpath,'_output.tif');

if frames == 3
    n=1;
    for i = 1:sections
        d1 = data(:,:,n);
        d2 = data(:,:,n+1);
        d3 = data(:,:,n+2);
        f1 = d1-d2; f2=d1-d3; f3=d2-d3;
        temp = f1.^2 + f2.^2 + f3.^2;
        temp = temp.^0.5;
        temp =temp-min(temp(:));
        output(:,:,i) = (65000.*(temp./max(temp(:))));
        imwrite(uint16(output(:,:,i)),svpath,'writemode','append');
        n=n+3;
    end

else
    disp('Too many frames...')
end


function TifStack = getNormTifStack(svpath)

    % Image loading constants
    edges = linspace(0,2^16,(2^8));
    PSF_edge = fspecial('gaussian',5,40);

    % Image reading loop
    TifStack = double(imread(svpath, 1));  
    for i = 1:27
        load = double(imread(svpath,i));
        [X, Y] = size(load);
        lim = min(X,Y);
        load=load(1:lim,1:lim);
        %load = load-min(load(:));
        %load = load./max(load(:));
        TifStack(:,:,i) = load;
    end
    
end
