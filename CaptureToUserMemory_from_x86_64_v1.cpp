#include <iostream>
#include <memory>
#include <thread>
#include <apps/Common/exampleHelper.h>
#include <common/minmax.h>
#include <mvIMPACT_CPP/mvIMPACT_acquire.h>
#include <mvIMPACT_CPP/mvIMPACT_acquire_GenICam.h>
#ifdef _WIN32
#   include <mvDisplay/Include/mvIMPACT_acquire_display.h>
using namespace mvIMPACT::acquire::display;
#   define USE_DISPLAY
#elif defined(linux) || defined(__linux) || defined(__linux__)
#   if defined(__x86_64__) || defined(__aarch64__) || defined(__powerpc64__) // -m64 makes GCC define __powerpc64__
using UINT_PTR = uint64_t;
#   elif defined(__i386__) || defined(__arm__) || defined(__powerpc__) // and -m32 __powerpc__
using UINT_PTR = uint32_t;
#   endif
#endif // #ifdef _WIN32

#ifdef BUILD_WITH_OPENCV_SUPPORT
// In order to build this application with OpenCV support there are 2 options:
// a) Use Visual Studio 2013 or later with OpenCV 2.4.13. With this version this example project already contains
//    project configurations 'Debug OpenCV' and 'Release OpenCV'. The only thing left to do in that case is to define
//    an environment variable 'OPENCV_DIR' that points to the top-level folder of the OpenCV 2.4.13 package. Afterwards
//    when restarting Visual Studio the application can be build with a simple OpenCV example.
//    - Different OpenCV versions can be used as well, but you need to state the OpenCV lib names matching your
//      version in the linker section then as the libraries contain the version in their name
// b) Older Visual Studio versions or Linux/gcc approaches must make sure that the include and lib paths to
//    OpenCV are set up correctly. However information on how to efficiently create OpenCV buffers can be extracted
//    from the code without compiling this application as well. Simply look for every occurrence of 'BUILD_WITH_OPENCV_SUPPORT'
//    in this source file.
#   include <opencv2/core/core.hpp>
#   include <opencv2/highgui/highgui.hpp>
#   include <opencv2/imgproc/imgproc.hpp>
#endif // #ifdef BUILD_WITH_OPENCV_SUPPORT

using namespace std;
using namespace mvIMPACT::acquire;
using namespace GenICam;

static bool s_boTerminated = false;

//-----------------------------------------------------------------------------
// the buffer we pass to the device driver must be aligned according to its requirements
// As we can't allocate aligned heap memory we will align it 'by hand'
class UserSuppliedHeapBuffer
//-----------------------------------------------------------------------------
{
    unique_ptr<char[]> pBuf_;
    int bufSize_;
    int alignment_;
public:
    explicit UserSuppliedHeapBuffer( int bufSize, int alignment ) : pBuf_(), bufSize_( bufSize ), alignment_( alignment )
    {
        if( bufSize_ > 0 )
        {
            pBuf_ = unique_ptr<char[]>( new char[bufSize_ + alignment_] );
        }
    }
    char* getPtr( void )
    {
        if( alignment_ <= 1 )
        {
            return pBuf_.get();
        }
        return reinterpret_cast<char*>( align( reinterpret_cast<UINT_PTR>( pBuf_.get() ), static_cast<UINT_PTR>( alignment_ ) ) );
    }
    int getSize( void ) const
    {
        return bufSize_;
    }
};

using CaptureBufferContainer = std::vector<shared_ptr<UserSuppliedHeapBuffer>>;

//-----------------------------------------------------------------------------
struct CaptureParameter
//-----------------------------------------------------------------------------
{
    Device* pDev;
#ifdef USE_DISPLAY
    shared_ptr<ImageDisplayWindow> pDisplayWindow;
#endif // #ifdef USE_DISPLAY
#ifdef BUILD_WITH_OPENCV_SUPPORT
    std::string             openCVDisplayTitle;
    std::string             openCVResultDisplayTitle;
#endif // #ifdef BUILD_WITH_OPENCV_SUPPORT
    FunctionInterface       fi;
    ImageRequestControl     irc;
    Statistics              statistics;
    bool                    boUserSuppliedMemoryUsed;
    bool                    boAlwaysUseNewUserSuppliedBuffers;
    int                     bufferSize;
    int                     bufferAlignment;
    int                     bufferPitch;
    CaptureBufferContainer  buffers;
    explicit CaptureParameter( Device* p ) : pDev{p}, fi{p}, irc{p}, statistics{p}, boUserSuppliedMemoryUsed{false},
        boAlwaysUseNewUserSuppliedBuffers{false}, bufferSize{0}, bufferAlignment{0}, bufferPitch{0}, buffers()
    {
#ifdef USE_DISPLAY
        // IMPORTANT: It's NOT save to create multiple display windows in multiple threads!!!
        pDisplayWindow = make_shared<ImageDisplayWindow>( "mvIMPACT_acquire sample, Device " + pDev->serial.read() );
#endif // #ifdef USE_DISPLAY
#ifdef BUILD_WITH_OPENCV_SUPPORT
        openCVDisplayTitle = string( "mvIMPACT_acquire sample, Device " + pDev->serial.read() + ", OpenCV display" );
        openCVResultDisplayTitle = openCVDisplayTitle + "(Result)";
#endif // #ifdef BUILD_WITH_OPENCV_SUPPORT
    }
    CaptureParameter( const CaptureParameter& src ) = delete;
    CaptureParameter& operator=( const CaptureParameter& rhs ) = delete;
};

void checkCaptureBufferAddress( const Request* const pRequest, bool boShouldContainUserSuppliedMemory, const CaptureBufferContainer& buffers );
int  createCaptureBuffer( FunctionInterface& fi, CaptureBufferContainer& buffers, const int bufferSize, const int bufferAlignment, const int bufferPitch, unsigned int requestNr );
int  createCaptureBuffers( FunctionInterface& fi, CaptureBufferContainer& buffers, const int bufferSize, const int bufferAlignment, const int bufferPitch );
void freeCaptureBuffer( FunctionInterface& fi, CaptureBufferContainer& buffers, unsigned int requestNr );
void freeCaptureBuffers( FunctionInterface& fi, CaptureBufferContainer& buffers );
void runLiveLoop( CaptureParameter& captureParams );

//-----------------------------------------------------------------------------
/// \brief This function checks whether a buffer returned from an acquisition into a
/// request that has been assigned a user supplied buffer really contains a buffer
/// pointer that has been assigned by the user.
void checkCaptureBufferAddress( const Request* const pRequest, bool boShouldContainUserSuppliedMemory, const CaptureBufferContainer& buffers )
//-----------------------------------------------------------------------------
{
    if( boShouldContainUserSuppliedMemory && ( pRequest->imageMemoryMode.read() != rimmUser ) )
    {
        cout << "ERROR: Request number " << pRequest->getNumber() << " is supposed to contain user supplied memory, but claims that it doesn't." << endl;
        return;
    }
    else if( !boShouldContainUserSuppliedMemory )
    {
        if( pRequest->imageMemoryMode.read() == rimmUser )
        {
            cout << "ERROR: Request number " << pRequest->getNumber() << " is supposed NOT to contain user supplied memory, but claims that it does." << endl;
        }
        return;
    }

    const void* const pAddr = pRequest->imageData.read();
    for( const auto& buffer : buffers )
    {
        if( pAddr == buffer->getPtr() )
        {
            // found the buffer that has been assigned by the user
            return;
        }
    }

    cout << "ERROR: A buffer has been returned, that doesn't match any of the buffers assigned as user memory in request number " << pRequest->getNumber() << "." << endl;
    cout << "Buffer got: 0x" << pAddr << endl;
    cout << "Buffers allocated:" << endl;
    const CaptureBufferContainer::size_type vSize = buffers.size();
    for( CaptureBufferContainer::size_type j = 0; j < vSize; j++ )
    {
        cout << "[" << j << "]: 0x" << reinterpret_cast<void*>( buffers[j]->getPtr() ) << endl;
    }
}

//-----------------------------------------------------------------------------
int createCaptureBuffer( FunctionInterface& fi, CaptureBufferContainer& buffers, const int bufferSize, const int bufferAlignment, const int /*bufferPitch*/, unsigned int requestNr )
//-----------------------------------------------------------------------------
{
    int functionResult = DMR_NO_ERROR;
    Request* pRequest = fi.getRequest( requestNr );
    shared_ptr<UserSuppliedHeapBuffer> pBuffer = make_shared<UserSuppliedHeapBuffer>( bufferSize, bufferAlignment );
    if( ( functionResult = pRequest->attachUserBuffer( pBuffer->getPtr(), pBuffer->getSize() ) ) != DMR_NO_ERROR )
    {
        cout << "An error occurred while attaching a buffer to request number " << requestNr << ": " << ImpactAcquireException::getErrorCodeAsString( functionResult ) << "." << endl;
        return -1;
    }
    buffers.push_back( pBuffer );
    return 0;
}

//-----------------------------------------------------------------------------
int createCaptureBuffers( FunctionInterface& fi, CaptureBufferContainer& buffers, const int bufferSize, const int bufferAlignment, const int bufferPitch )
//-----------------------------------------------------------------------------
{
    freeCaptureBuffers( fi, buffers );

    unsigned int requestCnt = fi.requestCount();
    for( unsigned int i = 0; i < requestCnt; i++ )
    {
        try
        {
            const int result = createCaptureBuffer( fi, buffers, bufferSize, bufferAlignment, bufferPitch, i );
            if( result != 0 )
            {
                freeCaptureBuffers( fi, buffers );
                return result;
            }
        }
        catch( const ImpactAcquireException& e )
        {
            freeCaptureBuffers( fi, buffers );
            return e.getErrorCode();
        }
    }
    return 0;
}

//-----------------------------------------------------------------------------
void freeCaptureBuffer( FunctionInterface& fi, CaptureBufferContainer& buffers, unsigned int requestNr )
//-----------------------------------------------------------------------------
{
    try
    {
        int functionResult = DMR_NO_ERROR;
        Request* pRequest = fi.getRequest( requestNr );
        if( pRequest->imageMemoryMode.read() == rimmUser )
        {
            if( ( functionResult = pRequest->detachUserBuffer() ) != DMR_NO_ERROR )
            {
                cout << "An error occurred while detaching a buffer from request number " << requestNr << " : " << ImpactAcquireException::getErrorCodeAsString( functionResult ) << "." << endl;
            }
        }
        const void* const pAddr = pRequest->imageData.read();
        CaptureBufferContainer::iterator it = find_if( buffers.begin(), buffers.end(), [pAddr]( const shared_ptr<UserSuppliedHeapBuffer>& buffer )
        {
            return pAddr == buffer->getPtr();
        } );
        if( it != buffers.end() )
        {
            buffers.erase( it );
        }
    }
    catch( const ImpactAcquireException& e )
    {
        cout << "An error occurred while changing the mode of request number " << requestNr << ": " << e.getErrorCodeAsString() << "." << endl;
    }
}

//-----------------------------------------------------------------------------
void freeCaptureBuffers( FunctionInterface& fi, CaptureBufferContainer& buffers )
//-----------------------------------------------------------------------------
{
    const unsigned int requestCnt = fi.requestCount();
    for( unsigned int i = 0; i < requestCnt; i++ )
    {
        freeCaptureBuffer( fi, buffers, i );
    }

    if( !buffers.empty() )
    {
        cout << "Error! The buffer container should be empty now but still contains " << buffers.size() << " elements!" << endl;
    }
}

//-----------------------------------------------------------------------------
void displayImage( CaptureParameter* pCaptureParameter, Request* pRequest )
//-----------------------------------------------------------------------------
{
#if !defined(USE_DISPLAY) && !defined(BUILD_WITH_OPENCV_SUPPORT)
    // suppress compiler warnings
    ( void )pRequest;
    ( void )pCaptureParameter;
#endif // #if !defined(USE_DISPLAY) && !defined(BUILD_WITH_OPENCV_SUPPORT)
#ifdef USE_DISPLAY
    pCaptureParameter->pDisplayWindow->GetImageDisplay().SetImage( pRequest );
    pCaptureParameter->pDisplayWindow->GetImageDisplay().Update();
#endif // #ifdef USE_DISPLAY
#ifdef BUILD_WITH_OPENCV_SUPPORT
    int openCVDataType = CV_8UC1;
    switch( pRequest->imagePixelFormat.read() )
    {
    case ibpfMono8:
        openCVDataType = CV_8UC1;
        break;
    case ibpfMono10:
    case ibpfMono12:
    case ibpfMono14:
    case ibpfMono16:
        openCVDataType = CV_16UC1;
        break;
    case ibpfMono32:
        openCVDataType = CV_32SC1;
        break;
    case ibpfBGR888Packed:
    case ibpfRGB888Packed:
        openCVDataType = CV_8UC3;
        break;
    case ibpfRGBx888Packed:
        openCVDataType = CV_8UC4;
        break;
    case ibpfRGB101010Packed:
    case ibpfRGB121212Packed:
    case ibpfRGB141414Packed:
    case ibpfRGB161616Packed:
        openCVDataType = CV_16UC3;
        break;
    case ibpfMono12Packed_V1:
    case ibpfMono12Packed_V2:
    case ibpfBGR101010Packed_V2:
    case ibpfRGB888Planar:
    case ibpfRGBx888Planar:
    case ibpfYUV422Packed:
    case ibpfYUV422_10Packed:
    case ibpfYUV422_UYVYPacked:
    case ibpfYUV422_UYVY_10Packed:
    case ibpfYUV422Planar:
    case ibpfYUV444Packed:
    case ibpfYUV444_10Packed:
    case ibpfYUV444_UYVPacked:
    case ibpfYUV444_UYV_10Packed:
    case ibpfYUV444Planar:
    case ibpfYUV411_UYYVYY_Packed:
        cout << "ERROR! Don't know how to render this pixel format (" << pRequest->imagePixelFormat.readS() << ") in OpenCV! Select another one e.g. by writing to mvIMPACT::acquire::ImageDestination::pixelFormat!" << endl;
        exit( 42 );
        break;
    }
    cv::Mat openCVImage( cv::Size( pRequest->imageWidth.read(), pRequest->imageHeight.read() ), openCVDataType, pRequest->imageData.read(), pRequest->imageLinePitch.read() );
    cv::imshow( pCaptureParameter->openCVDisplayTitle, openCVImage );
    // OpenCV event handling: you need this!
    cv::waitKey( 5 );
    // apply Canny Edge detection and display the result too
    cv::Mat edgesMat;
    switch( openCVDataType )
    {
    case CV_16UC3:
        cout << "This format seems to crash the Canny Edge detector. Will display the original image instead!" << endl;
        edgesMat = openCVImage;
        break;
    default:
        cv::Canny( openCVImage, edgesMat, 35.0, 55.0 );
        break;
    }
    cv::imshow( pCaptureParameter->openCVResultDisplayTitle, edgesMat );
    // OpenCV event handling: you need this!
    cv::waitKey( 5 );
#endif // #ifdef BUILD_WITH_OPENCV_SUPPORT
}

//-----------------------------------------------------------------------------
void liveLoop( CaptureParameter* pParameter )
//-----------------------------------------------------------------------------
{
    // Send all requests to the capture queue. There can be more than 1 queue for some devices, but for this sample
    // we will work with the default capture queue. If a device supports more than one capture or result
    // queue, this will be stated in the manual. If nothing is mentioned about it, the device supports one
    // queue only. This loop will send all requests currently available to the driver. To modify the number of requests
    // use the property mvIMPACT::acquire::SystemSettings::requestCount at runtime or the property
    // mvIMPACT::acquire::Device::defaultRequestCount BEFORE opening the device.
    TDMR_ERROR result = DMR_NO_ERROR;
    while( ( result = static_cast<TDMR_ERROR>( pParameter->fi.imageRequestSingle() ) ) == DMR_NO_ERROR ) {};
    if( result != DEV_NO_FREE_REQUEST_AVAILABLE )
    {
        cout << "'FunctionInterface.imageRequestSingle' returned with an unexpected result: " << result
             << "(" << ImpactAcquireException::getErrorCodeAsString( result ) << ")" << endl;
    }

    manuallyStartAcquisitionIfNeeded( pParameter->pDev, pParameter->fi );
    // run thread loop
    unsigned int cnt = {0};
    const unsigned int timeout_ms = {500};
    Request* pRequest = nullptr;
    // we always have to keep at least 2 images as the display module might want to repaint the image, thus we
    // cannot free it unless we have a assigned the display to a new buffer.
    Request* pPreviousRequest = nullptr;
    while( !s_boTerminated )
    {
        // wait for results from the default capture queue
        const int requestNr = pParameter->fi.imageRequestWaitFor( timeout_ms );
        pRequest = pParameter->fi.isRequestNrValid( requestNr ) ? pParameter->fi.getRequest( requestNr ) : 0;
        if( pRequest != nullptr )
        {
            if( pRequest->isOK() )
            {
                ++cnt;
                // here we can display some statistical information every 100th image
                if( cnt % 100 == 0 )
                {
                    cout << "Info from " << pParameter->pDev->serial.read()
                         << ": " << pParameter->statistics.framesPerSecond.name() << ": " << pParameter->statistics.framesPerSecond.readS()
                         << ", " << pParameter->statistics.errorCount.name() << ": " << pParameter->statistics.errorCount.readS()
                         << ", " << pParameter->statistics.captureTime_s.name() << ": " << pParameter->statistics.captureTime_s.readS()
                         << ", CaptureDimension: " << pRequest->imageWidth.read() << "x" << pRequest->imageHeight.read() << "(" << pRequest->imagePixelFormat.readS() << ")" << endl;
                }
                displayImage( pParameter, pRequest );
                checkCaptureBufferAddress( pRequest, pParameter->boUserSuppliedMemoryUsed, pParameter->buffers );
            }
            else
            {
                cout << "Error: " << pRequest->requestResult.readS() << endl;
            }
            if( pPreviousRequest != nullptr )
            {
                // this image has been displayed thus the buffer is no longer needed...
                pPreviousRequest->unlock();
                if( pParameter->boAlwaysUseNewUserSuppliedBuffers )
                {
                    // attach a fresh piece of memory
                    freeCaptureBuffer( pParameter->fi, pParameter->buffers, pPreviousRequest->getNumber() );
                    createCaptureBuffer( pParameter->fi, pParameter->buffers, pParameter->bufferSize, pParameter->bufferAlignment, pParameter->bufferPitch, pPreviousRequest->getNumber() );
                }
            }
            pPreviousRequest = pRequest;
            // send a new image request into the capture queue
            pParameter->fi.imageRequestSingle();
        }
        //else
        //{
        // Please note that slow systems or interface technologies in combination with high resolution sensors
        // might need more time to transmit an image than the timeout value which has been passed to imageRequestWaitFor().
        // If this is the case simply wait multiple times OR increase the timeout(not recommended as usually not necessary
        // and potentially makes the capture thread less responsive) and rebuild this application.
        // Once the device is configured for triggered image acquisition and the timeout elapsed before
        // the device has been triggered this might happen as well.
        // The return code would be -2119(DEV_WAIT_FOR_REQUEST_FAILED) in that case, the documentation will provide
        // additional information under TDMR_ERROR in the interface reference.
        // If waiting with an infinite timeout(-1) it will be necessary to call 'imageRequestReset' from another thread
        // to force 'imageRequestWaitFor' to return when no data is coming from the device/can be captured.
        // cout << "imageRequestWaitFor failed (" << requestNr << ", " << ImpactAcquireException::getErrorCodeAsString( requestNr ) << ")"
        //   << ", timeout value too small?" << endl;
        //}
    }
    manuallyStopAcquisitionIfNeeded( pParameter->pDev, pParameter->fi );
#ifdef USE_DISPLAY
    // stop the display from showing freed memory
    pParameter->pDisplayWindow->GetImageDisplay().RemoveImage();
#endif // #ifdef USE_DISPLAY
#ifdef BUILD_WITH_OPENCV_SUPPORT
    cv::destroyAllWindows();
#endif // #ifdef BUILD_WITH_OPENCV_SUPPORT
    // In this sample all the next lines are redundant as the device driver will be
    // closed now, but in a real world application a thread like this might be started
    // several times an then it becomes crucial to clean up correctly.

    // free the last potentially locked request
    if( pRequest != nullptr )
    {
        pRequest->unlock();
    }
    // clear all queues
    pParameter->fi.imageRequestReset( 0, 0 );
}

//-----------------------------------------------------------------------------
void runLiveLoop( CaptureParameter& parameter )
//-----------------------------------------------------------------------------
{
    s_boTerminated = false;
    thread myThread( liveLoop, &parameter );
    cin.get();
    s_boTerminated = true;
    myThread.join();
}


mvIMPACT::acquire::Device* getFirstDevice(const mvIMPACT::acquire::DeviceManager& devMgr)
{
    if( devMgr.deviceCount() == 0 ){
        std::cout << "No compliant device found!" << std::endl;
        return nullptr;
    }
    return devMgr[0];
}

int open(Device* pDev){
    cout << "opening camera" << endl;
    try{
        pDev->open();
        return 1;
    }
    catch( const ImpactAcquireException& e ){
        // this e.g. might happen if the same device is already opened in another process...
        cout << "An error occurred while opening device " << pDev->serial.read()
             << "(error code: " << e.getErrorCodeAsString() << ")." << endl;
        return 0;
    }
}

int first_camera_settings(Device* pDev){
    conditionalSetProperty( pDev->interfaceLayout, dilGenICam, true);
    if (pDev->acquisitionStartStopBehaviour.isValid()){
        conditionalSetProperty(pDev->acquisitionStartStopBehaviour, assbUser, true);
    }
    return 1;
}

int second_camera_settings(
//        Device* pDev,
        FunctionInterface* fi,
        ImageFormatControl* im_format_ctrl,
        AcquisitionControl* acq_ctrl,
        AnalogControl* analog_ctrl,
        DigitalIOControl* digital_ctrl,
        ChunkDataControl* chunk_data_ctrl,
        ImageProcessing* im_proc,
        ImageDestination* im_dest,
        BasicDeviceSettings* basic_device_settings){
    string settings_xml_path = "/opt/mvIMPACT_Acquire/apps/CaptureToUserMemory/CameraDefaultSettings.xml";
    if (!fi->loadSetting(
            settings_xml_path,sfDefault, sGlobal)){
        return 0;
    }
    //skipping checks
    //Camera Driver Version Check
    //Camera Firmware Version Check
    //Device Link Speed Check

    im_format_ctrl->pixelFormat.writeS( "BayerRG8" );
    im_format_ctrl->binningHorizontal.write(2);
    im_format_ctrl->binningVertical.write(2);

    acq_ctrl->exposureTime.write(15000.0);
    cout << "Resulting FrameRate: " << acq_ctrl->mvResultingFrameRate.read() << endl;
    acq_ctrl->acquisitionFrameRateEnable.write(bTrue);
    acq_ctrl->acquisitionFrameRate.write(40.0);

    analog_ctrl->balanceWhiteAuto.writeS("Off");

    digital_ctrl->lineSelector.writeS("Line4");
    digital_ctrl->mvLineDebounceTimeRisingEdge.write(5000);

    chunk_data_ctrl->chunkModeActive.write(bTrue);

    im_proc->whiteBalance.write(wbpUser1);
    im_proc->whiteBalanceCalibration.write(wbcmOff);
    WhiteBalanceSettings wbs = im_proc->getWBUserSetting(0);
    wbs.WBAoiMode.write(amFull);

    im_dest->pixelFormat.write(idpfAuto);
    basic_device_settings->imageRequestTimeout_ms.write(0);

    return 1;
}


//-----------------------------------------------------------------------------
int main()
//-----------------------------------------------------------------------------
{
    DeviceManager devMgr;
    Device* pDev = getFirstDevice( devMgr );
//    Device* pDev = getDeviceFromUserInput( devMgr );
    if( !pDev ){
        return 1;
    }
    if( !open(pDev) ){
        return 1;
    }
    if( !first_camera_settings(pDev) ){
        return 1;
    }

    auto* fi = new FunctionInterface(pDev);
    // GeniCam
    auto* dev_ctrl = new DeviceControl(pDev);
    auto* im_form_ctrl = new ImageFormatControl( pDev );
    auto* acq_ctrl = new AcquisitionControl( pDev );
    auto* analog_ctrl = new AnalogControl( pDev );
    auto* digital_ctrl = new DigitalIOControl( pDev );
    auto* chunk_data_ctrl = new ChunkDataControl( pDev );

    auto* info = new Info( pDev );
    auto* basic_device_settings = new BasicDeviceSettings(pDev);

    auto* im_proc = new ImageProcessing(pDev);
    // we will get w_balance_settings in camera run time
    // auto* w_balance_settings = new WhiteBalanceSettings();
    auto* im_dest = new ImageDestination(pDev);
    auto* stats = new Statistics(pDev);

    // skipping ImageDisplay,  I think its a Windows thing

    if( !second_camera_settings(fi, im_form_ctrl, acq_ctrl, analog_ctrl, digital_ctrl,
                                chunk_data_ctrl, im_proc, im_dest, basic_device_settings) ){
        return 1;
    }


    CaptureParameter captureParams( pDev );

    runLiveLoop( captureParams );

    int requestNr = captureParams.fi.imageRequestWaitFor( -1 );

    // and end the application
    cout << "Press [ENTER] to end the application..." << endl;
    cin.get();
    captureParams.fi.imageRequestUnlock( requestNr );
    captureParams.fi.imageRequestReset( 0, 0 );
    return 0;
}
