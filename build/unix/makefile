#sudo apt-get install build-essential
#sudo apt-get install ffmpeg
#sudo apt-get install libav-tools
#sudo apt-get install libopencv-dev

CXX           = g++
CXXFLAGS      = -O2 -Wall -D__STDC_CONSTANT_MACROS `pkg-config --libs --cflags opencv`
LIBS          = -lm                     \
                -lpthread               \
                -lavutil                \
                -lavformat              \
                -lavcodec               \
                -lswscale
OBJS          = ../../src/ardrone/ardrone.o \
                ../../src/ardrone/command.o \
                ../../src/ardrone/config.o  \
                ../../src/ardrone/udp.o     \
                ../../src/ardrone/tcp.o     \
                ../../src/ardrone/navdata.o \
                ../../src/ardrone/version.o \
                ../../src/ardrone/video.o   \
                ../../src/main.o

# Change the variable NAME to desired name's of programs
NAME         = ARDroneMF6.a
$(NAME):     $(OBJS)
		$(CXX) $(OBJS) -o $(NAME) $(CXXFLAGS) $(LDFLAGS) $(LIBS) 

clean:;         rm -f *.o *~ $(NAME) $(OBJS)

install:        $(NAME)
		install -s $(NAME) $(DEST)
