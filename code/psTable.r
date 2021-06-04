tab <- dat%>%
    group_by(ps,treatment)%>%
    summarize(n=n(),percomplete=round(mean(complete)*100))%>%#,corr=round(cor(complete,p_complete),2))%>%#rmse=mean((complete-p_complete)^2)))%>%
    ungroup()%>%
    pivot_wider(names_from=treatment,values_from=c(n,percomplete))#,corr))

tab <- cbind(tab[1:17,],rbind(tab[18:33,],rep(NA,5)))

ttt <- kbl(tab,
           format='latex',
           booktabs=FALSE,
           col.names=rep(c("",rep(c("Trt","Ctl"),2)),2),
           caption="Sample sizes and \\% homework completion by treatment group in each of the 33 A/B tests.",
           label="info")%>%
    kable_styling()%>%
    column_spec(5,border_right=TRUE)%>%
    add_header_above(rep(c("Experiment"=1,"n"=2,"% Complete"=2),2))

sink('results/infoTab.tex')
cat(ttt)
sink()
