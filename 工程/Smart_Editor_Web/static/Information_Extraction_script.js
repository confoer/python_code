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

document.getElementById('upload-form').addEventListener('submit', function (e)
{
    e.preventDefault();
    var formData = new FormData(this);
    fetch('/upload_data',{
        method: 'POST',
        body: formData
        })
        .then(response => response.json())
        .then(data => {document.getElementById('result').value = data.result;})
        .catch(error => console.error('Error:', error));
});