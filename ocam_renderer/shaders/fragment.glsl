#version 330 core

in vec3 vcolor;
flat in int frag_inst_id;

layout(location = 0) out vec3 color;
layout(location = 1) out int pix_inst_id;

void main(){
	color = vcolor;
	pix_inst_id = frag_inst_id;
}