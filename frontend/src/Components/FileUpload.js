// FileUpload.js
import React from 'react';

export default function FileUpload({ pdfId, input, setInput, handleSend, loading }) {
    return (
        <footer className="fixed bottom-12 left-0 right-0 mx-6 sm:mx-16 md:mx-32 font-inter font-medium">
            <div
                className={`relative ${pdfId ? 'shadow-lg rounded-lg' : ''}`}
                style={{
                    boxShadow: pdfId ? '0px 4px 10px rgba(0, 0, 0, 0.1)' : 'none',
                    border: pdfId ? '1px solid rgba(228, 232, 238, 1)' : 'none',
                }}
            >
                <input
                    type="text"
                    className={`w-full pr-20 py-4 px-4 sm:px-8 rounded-lg ${pdfId ? 'bg-white text-black' : 'bg-gray-100 text-gray-500'
                        }`}
                    placeholder="Send a message..."
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={(e) => {
                        if (e.key === 'Enter') {
                            e.preventDefault();
                            handleSend();
                        }
                    }}
                    disabled={!Boolean(pdfId)} //Enable Text input if pdf is uploaded
                />
                <button
                    className="absolute right-4 sm:right-10 top-1/2 transform -translate-y-1/2 bg-transparent hover:bg-transparent"
                    onClick={handleSend}
                    disabled={!Boolean(pdfId) || loading}  //Enable Text send if pdf is uploaded
                >
                    {loading ? (
                        <div
                            className="inline-block h-8 w-8 animate-spin rounded-full border-4 border-solid border-current border-r-transparent align-[-0.125em] motion-reduce:animate-[spin_1.5s_linear_infinite]"
                            role="status"
                        >
                            <span className="!absolute !-m-px !h-px !w-px !overflow-hidden !whitespace-nowrap !border-0 !p-0 ![clip:rect(0,0,0,0)]">
                                Loading...
                            </span>
                        </div> //Loaded while we get response from Backend
                    ) : (
                        <img src="/iconoir_send.svg" className="h-6 w-6 sm:h-10 sm:w-10" alt="Send icon" />
                    )}
                </button>
            </div>
        </footer>
    );
}
