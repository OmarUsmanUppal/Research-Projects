# Practice Code from Udemy MOOC https://www.udemy.com/machlearn1/learn/v4/overview 

library(ggplot2)
x<-c(2,5,1)
y<-c(6,4,9)
dat<-data.frame(x,y)
dat

ggplot()+geom_point(data=dat,aes(x=x,y=y),size=5,color="blue")

ggplot()+geom_point(data=dat,aes(x=x,y=y),size=10,color="forestgreen",shape="p")+
    scale_x_continuous(limits = c(0,15),breaks = seq(0,15,5))+
    scale_y_continuous(limits = c(0,15),breaks = seq(0,15,5))


x<-c(1,8)
y<-c(3,10)
dat<-data.frame(x,y)
dat

ggplot()+geom_line(data=dat, aes(x=x,y=y))

dat$x<-c(4,3)
dat$y<-c(13,5)
dat
ggplot()+geom_line(data=dat, aes(x=x,y=y))

x<-c(0,10)
y<-3*x+1
y
dat<-data.frame(x,y)
ggplot()+geom_line(data=dat, aes(x=x,y=y))


sample(1:10,100,replace = TRUE)
rnorm(100,50,10)
rnorm(100,50,90)

