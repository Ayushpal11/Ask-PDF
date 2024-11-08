const API_URL = process.env.REACT_APP_API_URL; //Backend Api url from environment variable

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
    // Create a new FormData object to hold the PDF ID and question
    const formData = new FormData();
    formData.append("pdf_id", pdfId); // Append the PDF ID
    formData.append("question", question); // Append the question

    try {
        // Send a POST request to ask a question about the PDF
        const response = await fetch(`${API_URL}/ask_question/`, {
            method: "POST",
            body: formData,
        });

        // Check if the response is OK
        if (!response.ok) {
            throw new Error(`Error asking question: ${response.statusText}`);
        }

        // Return the response data as JSON
        return await response.json();
    } catch (error) {
        console.error(error);
        throw error; // Propagate the error to be handled by the calling function
    }
};