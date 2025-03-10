from OpenGL.GL import *
import numpy as np
import os
from .shader_loader import Shader
from .camera import camera_models, vertex_shader_models


class PointCloudRenderer:
    MULTISAMPLE_ENABLED = False

    class GLContext(object):
        pass

    def _delete_main_framebuffer(self):

        buf_list = [self._main_fb, self._main_cb, self._main_db]
        if self.MULTISAMPLE_ENABLED:
            buf_list += [self._main_fb_ms, self._main_cb_ms, self._main_db_ms]
        buf_list = [x for x in buf_list if x is not None]
        glDeleteFramebuffers(len(buf_list), buf_list)

        self._main_fb = None
        self._main_cb = None
        self._main_db = None
        self._main_fb_ms = None
        self._main_cb_ms = None
        self._main_db_ms = None
        self._main_fb_dims = (None, None)

    def _configure_main_framebuffer(self):
        # If mismatch with prior framebuffer, delete it
        if (self._main_fb is not None and
                (self.viewport_width != self._main_fb_dims[0] or
                 self.viewport_height != self._main_fb_dims[1])):
            self._delete_main_framebuffer()

        # If framebuffer doesn't exist, create it
        if self._main_fb is None:
            # Generate standard buffer
            self._main_cb, self._main_db = glGenRenderbuffers(2)

            self._main_ib = glGenRenderbuffers(1)

            glBindRenderbuffer(GL_RENDERBUFFER, self._main_ib)
            glRenderbufferStorage(
                GL_RENDERBUFFER, GL_R32I,
                self.viewport_width, self.viewport_height
            )

            glBindRenderbuffer(GL_RENDERBUFFER, self._main_cb)
            glRenderbufferStorage(
                GL_RENDERBUFFER, GL_RGBA,
                self.viewport_width, self.viewport_height
            )

            glBindRenderbuffer(GL_RENDERBUFFER, self._main_db)
            glRenderbufferStorage(
                GL_RENDERBUFFER, GL_DEPTH_COMPONENT24,
                self.viewport_width, self.viewport_height
            )

            self._main_fb = glGenFramebuffers(1)
            glBindFramebuffer(GL_DRAW_FRAMEBUFFER, self._main_fb)
            glFramebufferRenderbuffer(
                GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                GL_RENDERBUFFER, self._main_cb
            )
            glFramebufferRenderbuffer(
                GL_DRAW_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
                GL_RENDERBUFFER, self._main_db
            )
            glFramebufferRenderbuffer(
                GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT1,
                GL_RENDERBUFFER, self._main_ib
            )

            if self.MULTISAMPLE_ENABLED:
                # Generate multisample buffer
                self._main_cb_ms, self._main_db_ms = glGenRenderbuffers(2)
                glBindRenderbuffer(GL_RENDERBUFFER, self._main_cb_ms)
                glRenderbufferStorageMultisample(
                    GL_RENDERBUFFER, 4, GL_RGBA,
                    self.viewport_width, self.viewport_height
                )
                glBindRenderbuffer(GL_RENDERBUFFER, self._main_db_ms)
                glRenderbufferStorageMultisample(
                    GL_RENDERBUFFER, 4, GL_DEPTH_COMPONENT24,
                    self.viewport_width, self.viewport_height
                )
                self._main_fb_ms = glGenFramebuffers(1)
                glBindFramebuffer(GL_DRAW_FRAMEBUFFER, self._main_fb_ms)
                glFramebufferRenderbuffer(
                    GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                    GL_RENDERBUFFER, self._main_cb_ms
                )
                glFramebufferRenderbuffer(
                    GL_DRAW_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
                    GL_RENDERBUFFER, self._main_db_ms
                )

            self._main_fb_dims = (self.viewport_width, self.viewport_height)

    def __init__(self, width, height):
        self.viewport_width = width
        self.viewport_height = height
        self._main_fb = None

    def __del__(self):
        pass

    def init_opengl(self):
        self._configure_main_framebuffer()
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, self._main_fb_ms if self.MULTISAMPLE_ENABLED else self._main_fb)
        glDrawBuffers([GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1])
        glClearColor(1.0, 1.0, 1.0, 0)
        glViewport(0, 0, self.viewport_width, self.viewport_height)
        glEnable(GL_DEPTH_TEST)
        glDepthMask(GL_TRUE)
        glDepthFunc(GL_LESS)
        glDepthRange(0.0, 1.0)

    def init_context(self, pointcloud, camera_mode, **camera_params):
        self.context = self.GLContext()

        self.shader = shader = Shader()
        dirname = os.path.dirname(os.path.abspath(__file__))
        shader.initShaderFromGLSL([os.path.join(dirname, "shaders/" + vertex_shader_models[camera_mode])],
                                  [os.path.join(dirname, "shaders/fragment.glsl")],
                                  [os.path.join(dirname, "shaders/geometry.glsl")])

        self.camera = camera_models[camera_mode](self.context, self.shader)
        self.camera.init_intrinsics(**camera_params)

        glverts = np.copy(pointcloud.vertices.astype(np.float32), order='C')
        glcolors = np.copy(pointcloud.colors[:, :3].astype(np.float32) / 255., order='C')
        glids = np.arange(len(glverts), dtype=np.int32)

        self.nglverts = len(glverts)

        self.context.vertexbuffer = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.context.vertexbuffer)
        glBufferData(GL_ARRAY_BUFFER, glverts.nbytes, glverts, GL_STATIC_DRAW)

        self.context.colorbuffer = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.context.colorbuffer)
        glBufferData(GL_ARRAY_BUFFER, glcolors.nbytes, glcolors, GL_STATIC_DRAW)

        self.context.idbuffer = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.context.idbuffer)
        glBufferData(GL_ARRAY_BUFFER, glids.nbytes, glids, GL_STATIC_DRAW)

    def locate_camera(self, quat, pose):
        self.camera.init_extrinsics(quat, pose)

    def get_image(self):
        width, height = self._main_fb_dims[0], self._main_fb_dims[1]

        if self.MULTISAMPLE_ENABLED:
            # Bind framebuffer and blit buffers
            glBindFramebuffer(GL_READ_FRAMEBUFFER, self._main_fb_ms)
            glBindFramebuffer(GL_DRAW_FRAMEBUFFER, self._main_fb)
            glBlitFramebuffer(
                0, 0, width, height, 0, 0, width, height,
                GL_COLOR_BUFFER_BIT, GL_LINEAR
            )
            glBlitFramebuffer(
                0, 0, width, height, 0, 0, width, height,
                GL_DEPTH_BUFFER_BIT, GL_NEAREST
            )
        glBindFramebuffer(GL_READ_FRAMEBUFFER, self._main_fb)
        glReadBuffer(GL_COLOR_ATTACHMENT0)
        color_buf = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)
        # GL_RGB => 3 components per pixel
        color = np.frombuffer(color_buf, np.uint8).reshape(height, width, 3)[::-1]

        glReadBuffer(GL_COLOR_ATTACHMENT1)
        ind_buf = glReadPixels(0, 0, width, height, GL_RED_INTEGER, GL_INT)
        indices = np.frombuffer(ind_buf, np.int32).reshape(height, width)[::-1]

        return color, indices

    def get_image_depth(self):
        width, height = self._main_fb_dims[0], self._main_fb_dims[1]

        if self.MULTISAMPLE_ENABLED:
            # Bind framebuffer and blit buffers
            glBindFramebuffer(GL_READ_FRAMEBUFFER, self._main_fb_ms)
            glBindFramebuffer(GL_DRAW_FRAMEBUFFER, self._main_fb)
            glBlitFramebuffer(
                0, 0, width, height, 0, 0, width, height,
                GL_COLOR_BUFFER_BIT, GL_LINEAR
            )
            glBlitFramebuffer(
                0, 0, width, height, 0, 0, width, height,
                GL_DEPTH_BUFFER_BIT, GL_NEAREST
            )
        glBindFramebuffer(GL_READ_FRAMEBUFFER, self._main_fb)
        glReadBuffer(GL_COLOR_ATTACHMENT0)
        color_buf = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)
        # GL_RGB => 3 components per pixel
        color = np.frombuffer(color_buf, np.uint8).reshape(height, width, 3)[::-1]
        depth_buf = glReadPixels(0, 0, width, height, GL_DEPTH_COMPONENT, GL_FLOAT)
        depth = np.frombuffer(depth_buf, np.float32).reshape(height, width)[::-1]

        return color, depth

    def request_color_async(self, pbo=None):
        width, height = self._main_fb_dims[0], self._main_fb_dims[1]
        if pbo is None:
            pbo = glGenBuffers(1)
            glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo)
            glBufferData(GL_PIXEL_PACK_BUFFER, (3 * width * height), None, GL_STREAM_READ)
        else:
            glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo)
        glBindFramebuffer(GL_READ_FRAMEBUFFER, self._main_fb)
        glReadBuffer(GL_COLOR_ATTACHMENT0)
        glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, 0)
        return pbo

    def get_requested_color(self, pbo, delete_pbo=True):
        width, height = self._main_fb_dims[0], self._main_fb_dims[1]
        glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo)
        bufferdata = glMapBuffer(GL_PIXEL_PACK_BUFFER, GL_READ_ONLY)
        data = np.frombuffer(ctypes.string_at(bufferdata, (3 * width * height)), np.uint8).reshape(height, width, 3)
        glUnmapBuffer(GL_PIXEL_PACK_BUFFER)
        glBindBuffer(GL_PIXEL_PACK_BUFFER, 0)
        if delete_pbo:
            glDeleteBuffers(1, [pbo])
        return data[::-1]

    def draw(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glClearBufferiv(GL_COLOR, 1, -1)

        self.shader.begin()
        self.camera.upload()

        glEnableVertexAttribArray(0)
        glBindBuffer(GL_ARRAY_BUFFER, self.context.vertexbuffer)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)

        glEnableVertexAttribArray(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.context.colorbuffer)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, None)

        glEnableVertexAttribArray(2)
        glBindBuffer(GL_ARRAY_BUFFER, self.context.idbuffer)
        glVertexAttribIPointer(2, 1, GL_INT, 0, None)

        glDrawArrays(GL_POINTS, 0, self.nglverts)

        glDisableVertexAttribArray(0)
        glDisableVertexAttribArray(1)
        self.shader.end()
