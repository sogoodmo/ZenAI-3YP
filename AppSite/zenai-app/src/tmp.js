import React, { useState, useRef, useEffect } from 'react';
import Webcam from 'react-webcam';
import './App.css';
import axios from 'axios';

const api = axios.create({
  baseURL: 'http://localhost:5000',
});

function App() {
  const [imageSrc, setImageSrc] = useState(null);
  const [imageReceived, setImageReceived] = useState(false);
  const webcamRef = useRef(null);
  const [capturing, setCapturing] = useState(false);

  useEffect(() => {
    let intervalId = null;

    if (capturing) {
      intervalId = setInterval(() => {
        const imageSrc = webcamRef.current.getScreenshot();
        setImageSrc(imageSrc);

        // Save the captured image to a file
        const link = document.createElement('a');
        link.download = 'captured.jpg';
        link.href = dataUrl;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);


        api.post('/webcam-frame', imageSrc, { headers: { 'Content-Type': 'image/jpeg' } })
          .then(response => {
            if (response.status === 200 && response.data !== 'error') {
              setImageReceived(response.data.image);
              console.log(response.data)
            } else {
              throw new Error('Failed to send webcam frame');
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
  }, [capturing]);

  const startCapture = () => {
    setCapturing(true);
  };

  const stopCapture = () => {
    setCapturing(false);
    setImageSrc(null);
    setImageReceived(false);
  };

  return (
    <div className="App">
      <div className="image-container">
        <Webcam
          audio={false}
          ref={webcamRef}
          screenshotFormat="image/jpeg"
          videoConstraints={{
            width: 640,
            height: 360,
            facingMode: 'user',
          }}
        />
        {imageSrc && <img src={imageSrc} alt="captured" />}
      </div>
      <div className="status-container">
        {imageReceived ? (
          <div>
            <span className="received-message">Received</span>
            <div className="image-received-container">
              <img src={`data:image/jpeg;base64,${imageReceived}`} alt="received" />
            </div>
          </div>
        ) : null}
      </div>
      <div className="button-container">
        <button onClick={startCapture}>Start capture</button>
        <button onClick={stopCapture}>Stop capture</button>
      </div>
    </div>
  );
}

export default App;