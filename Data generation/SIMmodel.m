clear all
clc
tic 

saveFlag = 0; % 1 or 0. will greatly increase calculation times
n_fil = 200; % number of filaments
k = 49* pi; % strip separation
theta =-1.4; % stripe angle
thickness = 2;
nSteps = 40; 


disp('Searching for GPU')
try
    testArr = zeros(512,512,'gpuArray');
    clear testArr
    disp('GPU device found');
    gpuFlag = 1;
catch
    disp('No useable GPU detected');
    gpuFlag = 0;
    
end

% Setup the 3D plot
coords =  linspace(-1,1,512);
[xv,yv,~] =  meshgrid(coords,coords,coords);
xv = xv*sin(theta);
yv = yv*cos(theta);
xv = xv+yv;
clear yv
xv = single(xv);
out =  zeros(512,512,512,'single');
% saveResult(xv,'coordinates')
% factor =  copy(out)



% Fill an array with filaments
for i = 1:n_fil

    textwaitbar(i, n_fil, 'Drawing filament');
    x1 = 5*( rand);
    y1 = 5*( rand);
    z1 = 5*( rand);

    x2 =  randi([2,3]);
    y2 =  randi([2,3]);
    z2 =  randi([2,3]);

    offX =  randi(400);
    offY =  randi(400);
    offZ =  randi(400);
    
    for t = 1:30000
        
        x = round(x1*(t/1000).^x2) +offX;
        y = round(y1*(t/1000).^y2) + offY;
        z = round(z1*(t/1000).^z2) + offZ;

        if x>thickness && x<(512-thickness) && y>thickness && y<(512-thickness) && z>thickness && z<(512-thickness)
            out(x-thickness:x+thickness,y-thickness:y+thickness,z-thickness:z+thickness) = 1;
        end
    end
end


saveResult(out,'filaments',1);

PSFem = zeros(512,512,512,'single'); % pre-allocate for speed
for page = 1:512
    textwaitbar(page, 512, 'Loading detection PSF')
    PSFem(:,:,page) = single(imread('D:\Work\Test datasets\OS-SIM\code\detection PSF.tif',page));
end
disp('Calculating detection OTF...')

% Just for fun lets find the best fft algorithm
fftw('swisdom',[]);
fftw('planner','patient');
OTFem = fftn(PSFem);
clear PSFem % We don't use it again

if saveFlag == 1
    saveResult(abs(OTFem),'detection OTF',1)
end


PSFex = zeros(512,512,512,'single'); % pre-allocate for speed
for page = 1:512
    textwaitbar(page, 512, 'Loading excitation PSF')
    PSFex(:,:,page) = single(imread('D:\Work\Test datasets\OS-SIM\code\excitation PSF.tif',page));
end

disp('Calculating excitation OTF...')
OTFex = (fftn(PSFex));
clear PSFex % We don't use it again
if saveFlag == 1
    saveResult(abs(OTFex),'excitation OTF',1)
end

finalOuptut = zeros(512,512,3*nSteps,'single');
if gpuFlag == 1 
    disp('Moving to GPU')

    temp = gpuArray(zeros(512,512,512,'single'));
    holder = gpuArray(zeros(512,512,512,'single'));
    
    for z = 1:nSteps
        out = circshift(out,6,3);
    for i = 1:3
        textwaitbar((z-1)*3+i, 3*nSteps, 'Calculating images')
        holder = gpuArray(xv); % Can only store 2 variables on GPU at a time
        temp = 1-sin(k*holder + i*2* pi/3);
    %     temp = (fftn(temp));
    %     if saveFlag == 1
    %     saveResult(log(abs(fftshift(temp)+1)),'pattern FT')
    %     end
    %     
    %     temp = temp.*OTFex;
    %     temp = fftshift(ifftn(temp));
        if saveFlag == 1
            saveResult(gather(temp),'pattern',0)
        end
        holder = gpuArray(out); 
        temp = holder.*temp;
        if saveFlag == 1
            saveResult(gather(temp),'Fluorescent response',0)
        end    
        clear holder % Clear one variable to make way for the fftn
        temp = fftn(temp);
        holder = gpuArray(OTFem); 
        temp = temp.*holder;
        clear holder
        temp =  (fftn(temp));
        if saveFlag == 1
            saveResult(gather(abs(temp)),'limited stack',0)
        end
    x = gather(fftshift(abs(temp(:,:,512))));
    x = x - min(x(:));
    x = x/max(x(:));
    finalOuptut(:,:,i+(z-1)*3) = imnoise(1e-3*x,'poisson');
    end
    end
else    
    for z = 1:nSteps
        out = circshift(out,4,3);
    for i = 1:3

        textwaitbar((z-1)*3+i, 3*nSteps, 'Calculating images')
        temp = 1-sin(k*xv + i*2* pi/3);
    %     temp = (fftn(temp));
    %     if saveFlag == 1
    %     saveResult(log(abs(fftshift(temp)+1)),'pattern FT')
    %     end
    %     
    %     temp = temp.*OTFex;
    %     temp = fftshift(ifftn(temp));
    %     if saveFlag == 1
    %         saveResult(temp,'pattern')
    %     end
        temp = out.*temp;
        if saveFlag == 1
            saveResult(gather(temp),'Fluorescent response',0)
        end    

        temp = fftn(temp);
        temp = temp.*OTFem;
        temp = (fftn(temp));
        if saveFlag == 1
            saveResult(gather(abs(temp)),'limited stack',0)
        end

        x = gather(fftshift(abs(temp(:,:,512))));
        x = x/max(x(:));
        x = x - min(x(:));
        finalOuptut(:,:,i+(z-1)*3) = imnoise(1e-3*x,'poisson') ;
    end
    end
end
    
saveResult(finalOuptut,'final image',1)

disp('Done')
toc 
clear all


function [] = saveResult(x,name,progress)
    x = x - min(x(:));
    x = 65535 * x/max(x(:));
    x = uint16(x);
    path = strcat(name,'.tif');
    if isfile(path)
     % File exists.
    delete(path);
    end
    
    msg = strcat('Saving image: ',path);
    frames = size(x,3);
    if frames >50
        writeFast(x,path,progress)
    else
        for p =1:frames
            if progress == 1
            textwaitbar(p, frames, msg);
            end
            imwrite(x(:,:,p),path,'writemode','append');
        end
    end
end
function [] = writeFast(x,path,progress)
    msg = strcat('Saving image: ',path);
    fTIF = Fast_Tiff_Write(path,0.125,0);
	for page =1:size(x,3)
        if progress == 1
        textwaitbar(page, size(x,3), msg);
        end
        fTIF.WriteIMG(x(:,:,page));
    end
    fTIF.close;
end
function textwaitbar(i, n, msg)
% A command line version of waitbar.
% Usage:
%   textwaitbar(i, n, msg)
% Input:
%   i   :   i-th iteration.
%   n   :   total iterations.
%   msg :   text message to print.
%
% Date      : 05/23/2019
% Author    : Xiaoxuan He   <hexxx937@umn.edu>
% Institute : University of Minnesota
%
% Previous percentage number.
persistent i_prev_prct;
% Current percentage number.
i_prct = floor(i ./ n * 100);
% Print message when counting starts.
if isempty(i_prev_prct) || i_prct < i_prev_prct
    i_prev_prct = 0;
    S_prev = getPrctStr(i_prev_prct);
    
    fprintf('%s: %s',msg, S_prev);
end
% Print updated percentage.
if i_prct ~= i_prev_prct
    S_prev = getPrctStr(i_prev_prct);
    fprintf(getBackspaceStr(numel(S_prev)));
    
    S = getPrctStr(i_prct);
    fprintf('%s', S);
    
    i_prev_prct = i_prct;
end
% Clear percentage variable.
if i_prct == 100
    fprintf(' Done.\n');
    clear i_prev_prct;
end
end
function S = getPrctStr(prct)
S = sprintf('%d%%  %s',prct,getDotStr(prct));
if prct < 10
    S = ['  ',S];
elseif prct < 100
    S = [' ',S];
end
end
function S = getDotStr(prct)
S = repmat(' ',1,10);
S(1:floor(prct/10)) = '.';
S = ['[',S,']'];
end
function S = getBackspaceStr(N)
S = repmat('\b',1,N);
end

