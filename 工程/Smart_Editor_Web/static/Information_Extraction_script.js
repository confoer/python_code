document.addEventListener('DOMContentLoaded', function ()
{
    var buttons = document.querySelectorAll('.action-button');
    buttons.forEach(function (button)
    {
        button.addEventListener('click', function ()
        {
            var action = this.id; 
            var resultTextarea = document.getElementById('result'); 
            fetch('/Information_Extraction',
                {
                method: 'POST',
                headers:
                {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ action: action })
            })
                .then(response => response.json())
                .then(data => {
                    resultTextarea.value = data.result;
                })
                .catch(error => console.error('Error:', error));
        });
    });
});

function handleFormSubmit(event) {
    event.preventDefault();
    var formData = new FormData(document.getElementById('upload-form'));

    fetch('/Upload_data', {
        method: 'POST',
        body: formData
    })
        .then(response => {
            if (response.ok) {
                return response.text();
            }
            throw new Error('Network response was not ok.');
        })
        .then(data => {
            console.log(data);
            alert(data);
        })
        .catch(error => console.error('Error:', error));
}