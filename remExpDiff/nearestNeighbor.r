### mahal method
library(parallel)
library(ggplot2)
K <- 5

dist <- function(x1,remX,sigNegSqrt){
    diff <- sweep(remX,2,x1)
    part2 <- sigNegSqrt%*%t(diff)
    colSums(part2^2)
}

distK <- function(x,k,remX,sigNegSqrt){
    ddd <- dist(x,remX,sigNegSqrt)
    cat('.')
    sort(ddd)[1:k]
}


mahalDists <- function(remStud,exStud,K=5,cols,samp=0){
    remX <- as.matrix(remStud[,cols])

    remMeans <- colMeans(remX)
    remX <- scale(remX, center=remMeans,scale=FALSE)

    exX <- as.matrix(exStud[,colnames(remX)])
    exX <- scale(exX,center=remMeans,scale=FALSE)

    SVD <- svd(remX)

    sigNegSqrt <- SVD$v%*%diag(SVD$d^(-1))%*%t(SVD$v)*sqrt(nrow(remX)-1)

    if(samp){
        set.seed(613)
        remX <- remX[sample(1:nrow(remX),samp),]
    }

    
    print('remnant distances')
    print(Sys.time())
    kDistsRem <- mclapply(as.data.frame(t(remX)),distK,k=K+1,remX=remX,sigNegSqrt=sigNegSqrt,mc.cores=16)
    print('remnant distances finished')

    kDistsMeanRem <- vapply(kDistsRem,function(x) mean(x[-1]),1.1)

    print('rct distances')
    print(Sys.time())
    kdistsEx <-
        mclapply(
            as.data.frame(t(exX)),
            distK,k=K,remX=remX,sigNegSqrt=sigNegSqrt,mc.cores=16)

    kDistsMeanEx <- lapply(unique(exStud$experiment),
                           function(ex)
                               vapply(kdistsEx[which(exStud$experiment==ex)],mean,1.1)
                           )

    names(kDistsMeanEx) <- as.character(unique(exStud$experiment))
    
    list(remDists=kDistsMeanRem,exDists=kDistsMeanEx)
}

makePlotDat <- function(remDists,exDists,Max=NULL,crosswalk=NULL){

    nExp <- length(exDists)
    if(is.null(crosswalk)){
        exps <-  1:nExp    
        expNum <- lapply(exps,function(i) rep(i,length(exDists[[i]])))
    } else {
        expNum <- lapply(names(exDists),
                         function(x) if(x%in%crosswalk){
                                         rep(names(crosswalk)[crosswalk==x],length(exDists[[x]]))
                         } else rep(x,length(exDists[[x]])))
        
        exps <- sapply(expNum,function(x) x[1])
    }
    
    ggDat <- data.frame(dist=c(unlist(exDists),rep(remDists,nExp)),
                        ex=c(unlist(expNum),rep(exps,each=length(remDists))),
                        exRem=c(rep('RCT',length(unlist(exDists))),rep('Remnant',length(remDists)*nExp))
                        )
    ggDat <- within(ggDat,logDist <- log(ifelse(dist<exp(-10),exp(-10),dist)))
    
    if(!is.null(Max)) ggDat <- ggDat[ggDat$dist<=Max,]

    ggDat
}

plots <- function(ggDat){
    
    logPlot <- ggplot(ggDat,aes(x=logDist,color=exRem))+
        geom_density()+facet_wrap(~ex)+ggtitle('Log Distances')+xlab('Log Distance')+theme(legend.pos='bottom')

    sqrtPlot <- ggplot(ggDat,aes(x=sqrt(dist),color=exRem))+
        geom_density()+facet_wrap(~ex)+ggtitle('Distances')+xlab('Distance')+theme(legend.pos='bottom')


    violin <- ggplot(ggDat,aes(x=exRem,y=logDist,fill=exRem))+
        geom_violin(trim=FALSE)+geom_boxplot(width=0.1)+facet_wrap(~ex)+labs(x=NULL,y='Log Distance')+theme(legend.pos='bottom')

    box <- ggplot(ggDat,aes(x=exRem,y=logDist,fill=exRem))+
        geom_boxplot()+facet_wrap(~ex)+labs(x=NULL,y='Log Distance')+theme(legend.pos='bottom')

    
list(logPlot=logPlot,sqrtPlot=sqrtPlot,violin=violin,box=box)
}

plotAllDat <- function(study1,study2=NULL,Max1=NULL,Max2=NULL,crosswalk=NULL){
    ggDat <- with(study1,makePlotDat(remDists,exDists,Max=Max1,crosswalk=crosswalk))

    if(!is.null(study2)){
        ggDat2 <- with(study2,makePlotDat(remDists,exDists,Max=Max2,crosswalk=crosswalk))
        if(length(intersect(ggDat2$ex,ggDat$ex))){
            if(is.numeric(ggDat2$ex)) ggDat2$ex <- ggDat2$ex+max(ggDat$ex) else warning('overlapping exp names')
        }
        ggDat <- rbind(ggDat,ggDat2)
    }

    ggDat
}


plotAll <- function(study1,study2,Max1=NULL,Max2=NULL,crosswalk=NULL){
    ggDat <- with(study1,makePlotDat(remDists,exDists,Max=Max1,crosswalk=crosswalk))

    if(!missing(study2)){
        ggDat2 <- with(study2,makePlotDat(remDists,exDists,Max=Max2,crosswalk=crosswalk))
        if(length(intersect(ggDat2$ex,ggDat$ex))){
            if(is.numeric(ggDat2$ex)) ggDat2$ex <- ggDat2$ex+max(ggDat$ex) else warning('overlapping exp names')
        }
        ggDat <- rbind(ggDat,ggDat2)
    }

    if(is.character(ggDat$ex)){
        exNum <- as.numeric(unique(ggDat$ex))
        if(all(!is.na(exNum)))
            ggDat$ex <- factor(ggDat$ex,levels=unique(ggDat$ex)[order(exNum)])
    }
    
    plots(ggDat)
}
    
#### get the experiment numbers from the paper
load('../results/fullres.RData')
crosswalk <- sapply(fullres,function(x) attributes(x)$psid)


############ study 1
load('../results/remStudAvgStudy1.RData')
load('../results/exStudAvgStudy1.RData')

study1 <- mahalDists(remStud,exStud,cols=2:18)#,samp=2000)

save(study1,file='study1.RData')

if(is.null(names(study1$exDists))) names(study1$exDists) <- unique(exStud$experiment)

###
range(unlist(study1$exDists))
range(study1$remDists) ### whoa 2000. outlier?
quantile(study1$remDists)
###           0%          25%          50%          75%         100% 
###    0.0000000    0.1559267    0.9239299    2.9534364 2007.5994603 
mean(study1$remDists<max(unlist(study1$exDists))) ### 0.999
sum(study1$remDists>max(unlist(study1$exDists))) ### 4

MM <- max(unlist(study1$exDists))

#study1plots <- plotAll(study1,Max1=MM,crosswalk=crosswalk)

#ggsave('logDistStudy1.jpg',plot=study1plots$logPlot,width=6.5,height=9,units='in')
#ggsave('sqrtDistStudy1.jpg',plot=study1plots$sqrtPlot,width=6.5,height=9,units='in')


############ study 2
load('../results/remStudAvgStudy2.RData')
load('../results/exStudAvgStudy2.RData')

exStud$experiment <- exStud$sequence_id_target

study2 <- mahalDists(remStud,exStud,cols=3:20)#,samp=2000)

save(study2,file='study2.RData')

if(is.null(names(study2$exDists))) names(study2$exDists) <- unique(exStud$experiment)

###
range(unlist(study2$exDists))
range(study2$remDists) 
MM2 <- max(unlist(study2$exDists))
sum(study2$remDists>MM2)
mean(study2$remDists>MM2)

#study2plots <- plotAll(study2,crosswalk=crosswalk,Max1=MM2)

#ggsave('logDistStudy2.jpg',plot=study2plots$logPlot,width=6.5,height=9,units='in')
#ggsave('sqrtDistStudy2.jpg',plot=study2plots$sqrtPlot,width=6.5,height=9,units='in')

#plotTogether <- plotAll(study1,study2,Max1=MM,Max2=MM2,crosswalk=crosswalk)
#ggsave('violinPlots.jpg',plot=plotTogether$violin,width=6.5,height=9,units='in')

rebarImp <- crosswalk
names(rebarImp) <- vapply(names(rebarImp),function(nn) round(100*fullres[[nn]]['rebar','improvement'],2),1.1)

#plotsByRebarImp <- plotAll(study1,study2,Max1=MM,crosswalk=rebarImp)
#ggsave('violinPlotsByRebarImp.jpg',plot=plotsByRebarImp$violin,width=6.5,height=9,units='in')
#ggsave('boxPlotsByRebarImp.jpg',plot=plotsByRebarImp$box,width=6.5,height=9,units='in')

#### boxplots orderd by Rebar, with experiment numbersgg
ggDat <- plotAllDat(study1,study2,Max1=MM,crosswalk=crosswalk)

#### take out RCT with no covariates
ggDat <- subset(ggDat,as.numeric(ex)<100)
ggDat$ex <- factor(ggDat$ex,levels=names(crosswalk)[order(as.numeric(names(rebarImp)))])
ggDat$exRemAbv <- ifelse(ggDat$exRem=='RCT','RCT','Rem.')
box <- ggplot(ggDat,aes(x=exRemAbv,y=logDist,fill=exRem))+
        geom_boxplot()+facet_wrap(~ex)+labs(x=NULL,y='Log Distance',fill=NULL)+theme(legend.pos='bottom')
ggsave('boxPlotsOrdNamed.pdf',plot=box,width=6,height=7.5,units='in')



### means?
#means1 <- data.frame(ps=names(study1$exDists),meanDist=vapply(study1$exDists,mean,1.1),
#                     rebarImp=vapply(names(study1$exDists), function(nn) fullres[[names(crosswalk)[crosswalk==nn]]]['rebar','improvement'],1.1),
#                     reloopImp=vapply(names(study1$exDists), function(nn) fullres[[names(crosswalk)[crosswalk==nn]]]['strat1','improvement'],1.1))

#with(means1,plot(meanDist,rebarImp))
#with(means1,abline(lm(rebarImp~meanDist)))
#with(means1,plot(meanDist,reloopImp))
#with(means1,abline(lm(reloopImp~meanDist)))


#means2 <- data.frame(ps=names(study2$exDists[-7]),meanDist=vapply(study2$exDists[-7],mean,1.1),
#                     rebarImp=vapply(names(study2$exDists[-7]), function(nn) fullres[[names(crosswalk)[crosswalk==nn]]]['rebar','improvement'],1.1),
 #                    reloopImp=vapply(names(study2$exDists[-7]), function(nn) fullres[[names(crosswalk)[crosswalk==nn]]]['strat1','improvement'],1.1))

#with(means2,plot(meanDist,rebarImp))
#with(means2,abline(lm(rebarImp~meanDist)))
#with(means2,plot(meanDist,reloopImp))
#with(means2,abline(lm(reloopImp~meanDist)))
