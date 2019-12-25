#! /bin/make
##############################################
# Makefile for the 'CaptureToUserMemory' sample
##############################################
#-------------------------------------------
ifndef MVIMPACT_ACQUIRE_SOURCE_DIR
    ifndef MVIMPACT_ACQUIRE_DIR
        ROOT_PATH=../..
    else
        ROOT_PATH=$(MVIMPACT_ACQUIRE_DIR)/apps_test
    endif
else
    ROOT_PATH=$(MVIMPACT_ACQUIRE_SOURCE_DIR)/apps_test
endif

#-------------------------------------------
BINARY_NAME=CaptureToUserMemory
MODULE_NAME=CaptureToUserMemory.legacy

#-------------------------------------------
# OOBJ =										\
OOBJ = lalala.o										\

#-------------------------------------------
vpath %.cpp ..

#-------------------------------------------
USR_INCLS =					\
# 	`pkg-config opencv --cflags --libs` 	\
#-------------------------------------------
USR_LIBS =		\
#	`pkg-config opencv --cflags --libs` 	

#-------------------------------------------
CPPFLAGS_1 = 				\
	-pedantic				\
	-DMALLOC_TRACE			\
	-std=c++11		\
	-I/usr/local/include -I/usr/local/include/opencv -L/usr/local/lib -L../../lib/arm64	\
	-lopencv_highgui -lopencv_imgproc -lopencv_core \
	-lstdc++ -lrt 

#-------------------------------------------
include $(ROOT_PATH)/Makefile.samp.inc

