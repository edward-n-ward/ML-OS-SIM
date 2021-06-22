clear all
close all
a_num=1;% number of pattern orientations
p_num=3;% phase shift times for each pattern orientation

[files, path] = uigetfile('D:\*.tif','multiselect','on');
for imFile = 1:length(files) 
if isa(files,'char')
    filepath = strcat(path,files);
else
    filepath = strcat(path,files{imFile});
end

info = imfinfo(filepath);
n = length(info);
frames = n;

disp(['Processing stack with ' num2str(frames) ' frames.']);

path = [filepath(1:end-4),'_unrolled.tif'];
fTIF = Fast_Tiff_Write(path,0.125,0);

for f = 1:(a_num*p_num)
   result = imread(filepath,f);
   load(:,:,f) = result;
   fTIF.WriteIMG(result);
end
h = waitbar(0,'processing image...');

for f = (a_num*p_num+1):n
    waitbar(f/frames)
    f_num = mod(f-1,a_num*p_num)+1;
    load(:,:,f_num) = (imread(filepath,f));
    
    for ii=1:a_num*p_num
       fTIF.WriteIMG(load(:,:,ii));
    end    

end
fTIF.close;
close(h);
end