#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(binding = 0) uniform UniformData
{
    mat4 projection_view;
    vec4 light_direction;
    vec4 camera_position;
}
uniform_data;

layout(push_constant) uniform PushConstantData
{
    mat4 model;
}
push_constant_data;

layout(location = 0) in  vec3 local_position;
layout(location = 1) in  vec3 local_normal;
layout(location = 2) in  vec2 uv_coord;
layout(location = 0) out vec3 world_position;
layout(location = 1) out vec3 world_normal;

void main()
{
    world_position = vec3(push_constant_data.model  * vec4(local_position, 1.0));
    world_normal   = mat3(push_constant_data.model) *      local_normal;
    gl_Position    = uniform_data.projection_view   * vec4(world_position, 1.0);
}
