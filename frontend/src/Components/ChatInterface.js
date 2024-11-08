// ChatInterface.js
import React, { useRef, useEffect } from 'react';

export default function ChatInterface({ messages }) {
    const messageEndRef = useRef(null);

    useEffect(() => {
        messageEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages]);

    return (
        <main className="flex-grow my-10 mb-20 mx-6 sm:mx-16 md:mx-32 overflow-y-auto scrollbar-hide">
            {messages.map((message, index) => (
                <div key={index} className="flex items-start space-x-4 mb-8 sm:mb-14">
                    <div
                        className={`w-10 h-10 sm:w-12 sm:h-12 rounded-full flex items-center justify-center text-white ${message.sender === 'user' ? 'bg-purple-400' : 'bg-green-400'
                            }`}
                    >
                        {message.sender === 'user' ? (
                            message.avatar
                        ) : (
                            <img src="/AI_logo.png" alt="AI Avatar" className="w-full h-full rounded-full" />
                        )}
                    </div>
                    <div className="flex-1">
                        <p className="text-gray-700 font-inter font-medium text-sm sm:text-base">{message.content}</p>
                    </div>
                </div>
            ))}
            <div ref={messageEndRef} />
        </main>
    );
}
