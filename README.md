# cudaRaytracer

This is just a project I made to understand better how raytracing works and to study a bit of vector math.<br/>
It is probably not made in the most efficient way possible and is NVIDIA-Cuda only, since the point of this project is just to learn how it works.

Also, all the graphics code is entirely written in Cuda without using any graphics API, and OpenGL is just used to obtain the device pointer to the pixel buffer, which the Cuda code renders the scene into before it is finally displayed as a quad texture.

**Here's some examples:**

![Image 1](https://github.com/user-attachments/assets/b13bd13b-9cd3-474c-9e30-b558f10b744e)
![Image 2](https://github.com/user-attachments/assets/9c5eed53-7ec0-4705-bf60-21ea8067bec1)
