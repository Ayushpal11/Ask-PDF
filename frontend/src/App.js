import React, { useState, useEffect } from 'react';
import { uploadPDF, askQuestion } from './api';
import Header from './Components/Header';
import ChatInterface from './Components/ChatInterface';
import FileUpload from './Components/FileUpload';
export default function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [pdfName, setPdfName] = useState('');
  const [pdfId, setPdfId] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const savedPdfId = localStorage.getItem('pdfId');
    const savedPdfName = localStorage.getItem('pdfName');
    const savedMessages = JSON.parse(localStorage.getItem('messages')) || [];
    if (savedPdfId && savedPdfName) {
      setPdfId(savedPdfId);
      setPdfName(savedPdfName);
      setMessages(savedMessages);
    }
  }, []);

  useEffect(() => {
    localStorage.setItem('pdfId', pdfId || '');
    localStorage.setItem('pdfName', pdfName || '');
    localStorage.setItem('messages', JSON.stringify(messages));
  }, [pdfId, pdfName, messages]);

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (file) {
      try {
        const response = await uploadPDF(file);
        if (response && response.pdf_id && response.filename) {
          setPdfName(response.filename);
          setPdfId(response.pdf_id);
          setMessages([]);
        } else {
          alert('Upload failed: Invalid response from server.');
        }
      } catch (error) {
        console.error('Error uploading PDF:', error);
        alert('Upload failed. Please try again.');
      }
    }
  };

  const handleSend = async () => {
    if (input.trim() && pdfId) {
      const userMessage = { sender: 'user', content: input, avatar: 'S' };
      setMessages([...messages, userMessage]);
      setInput('');
      setLoading(true);
      try {
        const response = await askQuestion(pdfId, input);
        if (response && response.pdf_id && response.answer) {
          const aiMessage = { sender: 'ai', content: response.answer };
          setMessages((prevMessages) => [...prevMessages, aiMessage]);
        } else {
          alert('Failed to retrieve answer.');
        }
      } catch (error) {
        console.error('Error fetching answer:', error);
        alert('PDF should have only 2000 Characters. Please upload different PDF');
      } finally {
        setLoading(false);
      }
    }
  };

  const handleReset = () => {
    setPdfName('');
    setPdfId(null);
    setMessages([]);
    setInput('');
    setLoading(false);
    localStorage.removeItem('pdfId');
    localStorage.removeItem('pdfName');
    localStorage.removeItem('messages');
  };

  return (
    <div className="flex flex-col h-screen">
      <Header pdfName={pdfName} handleFileUpload={handleFileUpload} handleReset={handleReset} />
      <ChatInterface messages={messages} />
      <FileUpload
        pdfId={pdfId}
        input={input}
        setInput={setInput}
        handleSend={handleSend}
        loading={loading}
      />
    </div>
  );
}
