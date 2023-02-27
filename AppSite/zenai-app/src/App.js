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
  const [lastTime, setLastTime] = useState(-1); 
  const [poseError, setPoseError] = useState('');
  const [poseFix, setPoseFix] = useState('');
  const [poseHeldTime, setPoseHeldTime] = useState(1);
  const [lastPose, setLastPose] = useState('')
  const [lastPoseError, setLastPoseError] = useState(null);
  const [feedbackList, setFeedbackList] = useState(null)
  const [difficulty, setDifficulty] = useState(10); 


  useEffect(() => {
    // Initialize local variables
    let intervalId = null;
    
    // Check if we are capturing from webcam or selected video
    if (capturing && !selectedVideo && useWebcam) {
      // Get webcam and start capturing frames
      const webcam = videoRef.current;
      intervalId = setInterval(() => {
        // Take screenshot of webcam and send frame to server
        const curImage = webcam.getScreenshot();
        setSentImage(curImage);
        api.post('/webcam-frame', { data: curImage, diff: difficulty })
          .then(response => {
            if (response.status === 200 && response.data !== 'error') {
              setImageReceived(response.data.image);
                const curPoseError = response.data.pose_error;
                setPoseError(curPoseError);
                setPoseFix(response.data.pose_fix)
                setFeedbackList(response.data.feedback)
                console.log(response.data.feedback)
                // TODO: Do something with classified pose
                if (lastPoseError === curPoseError) {
                  setPoseHeldTime(poseHeldTime + 2);
                  // console.log(`Same. Last == Cur: (should be true) ${lastPoseError, curPoseError} HeldTime: ${poseHeldTime}`);
                } else {
                  setPoseHeldTime(1);
                  setLastPoseError(curPoseError);
                  // console.log(`Diff. Last == Cur (Should be flase): ${lastPoseError, curPoseError} HeldTime: ${poseHeldTime}`);
                }
                // console.log(poseHeldTime)
            } else {
              throw new Error('Failed to send webcam frame');
            }
          })
          .catch(error => {
            console.error(error);
          });
      }, 1000);
    } else if (capturing && selectedVideo) {
      // Get video and canvas for drawing
      const video = videoRef.current;
      const canvas = document.createElement('canvas');
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      // Start capturing frames
      intervalId = setInterval(() => {
        if (video.currentTime !== lastTime) {
          // Draw video frame to canvas and send frame to server
          canvas.getContext('2d').drawImage(video, 0, 0);
          const curImage = canvas.toDataURL('image/jpeg', 0.5);
          setSentImage(curImage);
          setLastTime(video.currentTime);
          api.post('/webcam-frame', { data: curImage, diff: difficulty })
            .then(response => {
              if (response.status === 200 && response.data !== 'error') {
                // Process response and update state
                setImageReceived(response.data.image);
                const curPoseError = response.data.pose_error;
                setPoseError(curPoseError);
                setPoseFix(response.data.pose_fix)
                setFeedbackList(response.data.feedback)
                console.log(response.data.feedback)
                // TODO: Do something with classified pose
                if (lastPoseError === curPoseError) {
                  setPoseHeldTime(poseHeldTime + 2);
                  // console.log(`Same. Last == Cur: (should be true) ${lastPoseError, curPoseError} HeldTime: ${poseHeldTime}`);
                } else {
                  setPoseHeldTime(1);
                  setLastPoseError(curPoseError);
                  // console.log(`Diff. Last == Cur (Should be flase): ${lastPoseError, curPoseError} HeldTime: ${poseHeldTime}`);
                }
                // console.log(poseHeldTime)
              } else {
                throw new Error('Failed to send video frame');
              }
            })
            .catch(error => {
              console.error(error);
            });
        }
      }, 100);
    } else {
      // Clear interval if we're not capturing
      clearInterval(intervalId);
    }
    // Clean up interval on unmount
    return () => clearInterval(intervalId);
  }, [capturing, useWebcam, lastTime, difficulty]);
  
  
  

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

  const HighlightedText = ({ text }) => {
    const regex = /\+(.*?)\+/g;
    const matches = text.match(regex);

    const parts = text.split(regex);
    return (
      <div>
        {parts.map((part, i) => {
          if (matches.includes(`+${part}+`)) {
            return (
              <span key={i} style={{ color: 'red', fontStyle: 'italic' }}>
                {part}
              </span>
            );
          } else {
            return <span key={i}>{part}</span>;
          }
        })}
      </div>
    );
  };

  const handleSliderChange = (e) => {
    setDifficulty(e.target.value);
  };



  return (
    <div className="App">
      <div className="image-container">
        <div className="preset-button-container">
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
          <div className="slider-container">
            <h2 className="slider-text">Pose Difficulty: {difficulty}</h2>
            <input type="range" min="1" max="10" value={difficulty} onChange={handleSliderChange} style={{ width: '100%' }} />
          </div>
        </div>
      </div>

      {imageReceived ? (
        <div>
          <div className="image-received-container">
            <img src={`data:image/jpeg;base64,${imageReceived}`} alt="received" />

          <div className="feedback-container">
            {poseError !== '' && feedbackList != null ? 
            (<div className='reformat'>
              <h1><HighlightedText text={feedbackList[0]} /></h1>
              <h1><HighlightedText text={feedbackList[1]} /></h1>
              <h1><HighlightedText text={feedbackList[2]} /></h1>
              {/* <h2>{feedbackList[1]}</h2>
              <h2>{feedbackList[2]}</h2> */}
            </div>)
            : null}

            {poseHeldTime > 2 && poseError !== '' ? 
                (<div>
                  <h1><a className='bigger'>Common Mistakes:</a></h1> 
                  <h1>{poseError}!</h1> <h1>{poseFix}</h1>
                </div>) 
            : null}
            </div>
          </div>
        </div>
      ) : null}
    </div>
  );
}

export default App;
