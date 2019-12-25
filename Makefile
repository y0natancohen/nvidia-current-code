#! /bin/make
OSTYPE := $(shell uname | cut -d _ -f 1 | tr [:upper:] [:lower:])
HOSTTYPE := $(shell uname -m)

#-------------------------------------------
ifeq ($(OSTYPE),cygwin)
	SUBDIRS=$(OSTYPE)
else
	ifeq ($(HOSTTYPE),i686)
		NATIVE=x86
	else
		ifeq ($(HOSTTYPE), aarch64)
			NATIVE=arm64
		else
			NATIVE=$(HOSTTYPE)
			ifeq ($(NATIVE),x86_64)
				EXTRA_TARGET=x86
			endif
		endif
	endif

	SUBDIRS = $(NATIVE) arm64
endif

#-------------------------------------------
.PHONY: $(SUBDIRS) all build new info cygwin clean armv7a armv6zk armv7ahf armv7axe armhf armsf arm64

#-------------------------------------------
all info:
	@for dir in $(SUBDIRS) ; do mkdir -p $$dir ; $(MAKE) -C $$dir -f ../Makefile.inc $@ || exit $$?; done

#-------------------------------------------
native:
	@mkdir -p $(NATIVE)
	$(MAKE) -C $(NATIVE) -f ../Makefile.inc all || exit $$?

#-------------------------------------------
x86 x86_64 arm cygwin armv7a armv6zk armv7ahf armv7axe armhf armsf arm64:
	@mkdir -p $@
	$(MAKE) -C $@ -f ../Makefile.inc all || exit $$?

#-------------------------------------------
clean:
	@rm -rf $(SUBDIRS)

#-------------------------------------------
build new: clean
	@for dir in $(SUBDIRS) ; do mkdir -p $$dir; $(MAKE) -C $$dir -f ../Makefile.inc $@ || exit $$?; done

#-------------------------------------------
strip:
	@for dir in $(SUBDIRS);										\
	do															\
		if test -d $$dir;										\
		then													\
			$(MAKE) -C $$dir -f ../Makefile.inc $@ || exit $$?;	\
		fi;														\
	done
