const canvas = document.getElementById('gameCanvas');
canvas.width = 400;
canvas.height = 400;
const ctx = canvas.getContext('2d');

const gridSize = 20;
const tileCount = canvas.width / gridSize;

let snake = [
    {x: 10, y: 10}
];

let apple = {
    x: Math.floor(Math.random() * tileCount),
    y: Math.floor(Math.random() * tileCount)
};

let direction = 'right';

function draw() {
    ctx.fillStyle = '#f0f0f0';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    ctx.fillStyle = 'green';
    snake.forEach(segment => {
        ctx.fillRect(segment.x * gridSize, segment.y * gridSize, gridSize, gridSize);
    });

    ctx.fillStyle = 'red';
    ctx.fillRect(apple.x * gridSize, apple.y * gridSize, gridSize, gridSize);
}

function update() {
    const head = {...snake[0]};
    switch(direction) {
        case 'up':
            head.y--;
            break;
        case 'down':
            head.y++;
            break;
        case 'left':
            head.x--;
            break;
        case 'right':
            head.x++;
            break;
    }

    snake.unshift(head);

    if(head.x === apple.x && head.y === apple.y) {
        apple = {
            x: Math.floor(Math.random() * tileCount),
            y: Math.floor(Math.random() * tileCount)
        };
    } else {
        snake.pop();
    }
}

// 添加速度设置变量
const gameSpeed = 200; // 数值越小，速度越快

function gameLoop() {
    update();
    draw();
    setTimeout(gameLoop, gameSpeed);
}

document.addEventListener('keydown', (event) => {
    switch(event.key) {
        case 'ArrowUp':
            if(direction !== 'down') direction = 'up';
            break;
        case 'ArrowDown':
            if(direction !== 'up') direction = 'down';
            break;
        case 'ArrowLeft':
            if(direction !== 'right') direction = 'left';
            break;
        case 'ArrowRight':
            if(direction !== 'left') direction = 'right';
            break;
    }
});

gameLoop();