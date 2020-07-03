## Options example: 

--root C:/Users/charl/Desktop/SIMRec_0214_rndAll
--imageSize 512
--out C:/temp/0314_SIMRec_0214_rndAll_rcan_continued
--model rcan
--nch_in 9
--nch_out 1
--ntrain 2380
--ntest 20
--scale 1
--task simin_gtout
--n_resgroups 3
--n_resblocks 10
--n_feats 48
--lr 0.00001
--scheduler 5,0.5
--nepoch 60
--norm hist
--dataset fouriersim
--workers 6
--batchSize 1
--saveinterval 5
--plotinterval 5


## One-line command for training

python run.py --root C:/Users/charl/Desktop/SIMRec_0214_rndAll --imageSize 512 --out C:/temp/0314_SIMRec_0214_rndAll_rcan_continued --weights "C:/temp/0216_SIMRec_0214_rndAll_rcan_continued/prelim40.pth" --model rcan --nch_in 9 --nch_out 1 --ntrain 2380 --ntest 20 --scale 1 --task simin_gtout --n_resgroups 3 --n_resblocks 10 --n_feats 48 --lr 0.00001 --scheduler 5,0.5 --nepoch 60 --norm hist --dataset fouriersim --workers 6 --batchSize 1 --saveinterval 5 --plotinterval 5