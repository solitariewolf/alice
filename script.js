document.getElementById('send-button').addEventListener('click', function() {
    var userInput = document.getElementById('user-input').value;

    fetch('http://localhost:5000/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({message: userInput})
    })
    .then(response => response.json())
    .then(data => {
        var messageArea = document.getElementById('message-area');

        // Add the user's message to the chat area
        var userMessageDiv = document.createElement('div');
        userMessageDiv.textContent = 'Você: ' + userInput;
        messageArea.appendChild(userMessageDiv);

        // Add the chatbot's response to the chat area
        var botMessageDiv = document.createElement('div');
        
        var botMessageImage = document.createElement('img');
        botMessageImage.src = 'alice_icon.png'; // Substitua pelo caminho da imagem desejada
        botMessageImage.width = '35';
        botMessageImage.height = '35';
        botMessageDiv.appendChild(botMessageImage);
        
        var botResponse = document.createElement('span');
        botResponse.textContent = data.response;
        
        var aliceText = document.createTextNode(' Alice: ');
        botMessageDiv.appendChild(aliceText);
        
        botMessageDiv.appendChild(botResponse);

        messageArea.appendChild(botMessageDiv);
    });

    // Clear the user input
    document.getElementById('user-input').value = '';
});
document.addEventListener("DOMContentLoaded", function() {
  var userInput = document.getElementById("user-input");
  var sendButton = document.getElementById("send-button");

  userInput.addEventListener("keydown", function(event) {
    if (event.keyCode === 13) { // Verifica se a tecla pressionada é "Enter"
      event.preventDefault(); // Impede o comportamento padrão de inserir uma quebra de linha

      sendButton.click(); // Aciona o clique no botão "Enviar"
    }
  });
});

