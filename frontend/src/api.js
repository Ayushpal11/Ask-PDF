// import axios from "axios";

// const Api = axios.create({
//     baseURL: "http://localhost:8000", //for localhost connection with FastAPI backend 
//     // baseURL:".onrender.com"   //connection with deployed backend deployed FastAPI app 
// })

// export default Api

const API_URL = "http://127.0.0.1:8000" //Backend Api url from environment variable

export const uploadPDF = async (file) => {
    // Create a new FormData object to hold the file data
    const formData = new FormData();
    formData.append("file", file); // Append the file to the FormData object
    console.log(API_URL);
    console.log(FormData);
    try {
        // Send a POST request to upload the PDF
        const response = await fetch(`${API_URL}/upload_pdf/`, {
            method: "POST",
            body: formData,
        });

        // Check if the response is OK (status in the range 200-299)
        if (!response.ok) {
            throw new Error(`Error uploading PDF: ${response.statusText}`);
        }

        // Return the response data as JSON
        return await response.json();
    } catch (error) {
        console.error(error);
        throw error; // Propagate the error to be handled by the calling function
    }
};
export const askQuestion = async (pdfId, question) => {
    try {
        const response = await fetch(`${API_URL}/ask_question/`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                pdf_id: pdfId,
                question: question
            }),
        });

        if (!response.ok) {
            throw new Error(`Error asking question: ${response.statusText}`);
        }

        return await response.json();
    } catch (error) {
        console.error(error);
        throw error;
    }
};