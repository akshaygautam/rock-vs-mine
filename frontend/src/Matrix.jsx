// src/Matrix.js
import React, { useEffect, useState } from 'react';
import './Matrix.css'; // Ensure this file exists

const Matrix = () => {
  const [frequencyMap, setFrequencyMap] = useState({});
  const [selectedBoxes, setSelectedBoxes] = useState([]);

  useEffect(() => {
    fetch('http://localhost:8080/frequency-map')
      .then(response => response.json())
      .then(setFrequencyMap)
      .catch(error => console.error('Error fetching frequency map:', error));
  }, []);

  const size = Math.floor(Math.sqrt(Object.keys(frequencyMap).length));

  const handleClick = (key) => {
    if (!selectedBoxes.includes(key)) {
      setSelectedBoxes(prev => [...prev, key]);
    }
  };

  const renderFace = (value) => {
    switch (value) {
      case 'The object is a Mine':
        return '💣';
      case 'The object is a Rock':
        return '🪨';
      default:
        return '?';
    }
  };

  return (
    <div className="matrix-container">
      {Array.from({ length: size }, (_, rowIndex) => (
        <div key={rowIndex} className="matrix-row">
          {Array.from({ length: size }, (_, colIndex) => {
            const key = rowIndex * size + colIndex;
            const value = frequencyMap[key] || 'Unknown';
            const isRevealed = selectedBoxes.includes(key);

            return (
              <div
                key={colIndex}
                className={`matrix-box ${isRevealed ? value.toLowerCase() : 'hidden'}`}
                onClick={() => handleClick(key)}
                style={{ cursor: isRevealed ? 'not-allowed' : 'pointer' }}
              >
                <div className={`face ${isRevealed ? 'visible' : 'hidden'}`}>
                  {renderFace(value)}
                </div>
              </div>
            );
          })}
        </div>
      ))}
    </div>
  );
};

export default Matrix;
