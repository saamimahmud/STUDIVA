import React, { useState } from 'react';

function VivaPanel() {
  const [studentAnswer, setStudentAnswer] = useState('');
  const [score, setScore] = useState(null);

  const evaluate = async () => {
    const response = await fetch('http://localhost:5000/evaluate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        student_answer: studentAnswer,
        model_answer: "TCP ensures reliable and ordered data transmission by establishing a connection."
      })
    });
    const data = await response.json();
    setScore(data.similarity_score);
  };

  return (
    <div>
      <h2>Viva Evaluator</h2>
      <textarea 
        value={studentAnswer} 
        onChange={(e) => setStudentAnswer(e.target.value)} 
        placeholder="Paste your transcribed answer here"
      />
      <button onClick={evaluate}>Evaluate</button>
      {score && <p>Similarity Score: {score.toFixed(2)}</p>}
    </div>
  );
}

export default VivaPanel;
