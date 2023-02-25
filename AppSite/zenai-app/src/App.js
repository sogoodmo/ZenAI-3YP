import React, { useState, useRef, useEffect } from 'react';
import Webcam from 'react-webcam';
import './App.css';
import axios from 'axios';
import ChairExample from './server_videos/Chair.mp4'
import CobraExample from './server_videos/Cobra.mp4'
import WarriorIIExample from './server_videos/WarriorII.mp4'
import TreeExample from './server_videos/Tree.mp4'
import DowndogExample from './server_videos/Downdog.mp4'

const api = axios.create({
  baseURL: 'http://localhost:5000',
});

function App() {
  const [imageReceived, setImageReceived] = useState(false);
  const videoRef = useRef(null);
  const [capturing, setCapturing] = useState(false);
  const [selectedVideo, setSelectedVideo] = useState(null);
  const [useWebcam, setUseWebcam] = useState(true);
  const [sentImage, setSentImage] = useState(null);

  useEffect(() => {
    let intervalId = null;

    if (capturing && !selectedVideo && useWebcam) {
      const webcam = videoRef.current;
      intervalId = setInterval(() => {
        // console.log(timeSpent);

        const curImage = webcam.getScreenshot();
        setSentImage(curImage)
        api.post('/webcam-frame', { data: curImage })
          .then(response => {
            if (response.status === 200 && response.data !== 'error') {
              setImageReceived(response.data.image);
            } else {
              throw new Error('Failed to send webcam frame');
            }
          })
          .catch(error => {
            console.error(error);
          });
      }, 1000);
    } else if (capturing && selectedVideo) {
      const video = videoRef.current;
      const canvas = document.createElement('canvas');
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;

      intervalId = setInterval(() => {
        // console.log(timeSpent);

        canvas.getContext('2d').drawImage(video, 0, 0);
        const curImage = canvas.toDataURL('image/jpeg', 0.5);
        setSentImage(curImage)


        api.post('/webcam-frame', { data: curImage })
          .then(response => {
            if (response.status === 200 && response.data !== 'error') {
              setImageReceived(response.data.image);
              console.log(response.data);
            } else {
              throw new Error('Failed to send video frame');
            }
          })
          .catch(error => {
            console.error(error);
          });
      }, 100);
    } else {
      clearInterval(intervalId);
    }

    return () => clearInterval(intervalId);
  }, [capturing, selectedVideo, useWebcam]);

  const startCapture = () => {
    setCapturing(true);
  };

  const stopCapture = () => {
    setCapturing(false);
    setImageReceived(false);
  };
  
  const handleVideoChange = event => {
    const file = event.target.files[0];
    if (file) {
      setSelectedVideo(URL.createObjectURL(file));
      setUseWebcam(false);
    } 
  };

  const handleVideoChangePreset = (example) => {
    
    switch (example){
      case 'Chair':
        setSelectedVideo(ChairExample);
        break;
      case 'Tree':
        setSelectedVideo(TreeExample);
        break;
      case 'Cobra':
        setSelectedVideo(CobraExample);
        break;
      case 'WarriorII':
        setSelectedVideo(WarriorIIExample);
        break;
      case 'Downward Dog':
        setSelectedVideo(DowndogExample);
        break;
      default:
        setSelectedVideo(DowndogExample);
        break;
    }

    setUseWebcam(false);
  };

  const handleWebcamClick = () => {
    setSelectedVideo(null);
    setUseWebcam(true);
  };

  const renderButton = (label) => {
    return (
      <button onClick={() => handleVideoChangePreset(label)}>
        {label}
      </button>
    );
  };


  return (
    <div className="App">
      <div className="image-container">
        {selectedVideo ? (
          <video ref={videoRef} controls src={selectedVideo} />
        ) : (
          <Webcam
            audio={false}
            ref={videoRef}
            screenshotFormat="image/jpeg"
            videoConstraints={{
              width: 1280,
              height: 720,
              facingMode: 'user',
            }}
          />
        )}
        <div className="preset-button-container">
          <span className='preset-class'> Example Videos </span>
          {renderButton('Chair')}
          {renderButton('Cobra')}
          {renderButton('Downward Dog')}
          {renderButton('Tree')}
          {renderButton('WarriorII')}
        </div>
      </div>
        <div className="button-container">
          <button onClick={startCapture}>Start capture</button>
          <button onClick={stopCapture}>Stop capture</button>
          <button onClick={handleWebcamClick}>Capture webcam</button>
          <button className="button">
            <label>
              Upload Video
              <input type="file" accept="video/*" onChange={handleVideoChange} style={{ display: "none" }} />
            </label>.
          </button>
        </div>
      {imageReceived ? (
        <div>
          <span className="received-message"> Webcam Footage Returned From Server </span>
          <div className="image-received-container">
            <img src={`data:image/jpeg;base64,${imageReceived}`} alt="received" />
            </div>
        </div>
      ) : null}
    </div>
  );
}

export default App;
