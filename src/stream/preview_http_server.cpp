#include "preview_http_server.h"

#include <algorithm>
#include <iostream>
#include <sstream>
#include <cstring>

#ifdef USE_MONGOOSE_WS
#include "mongoose.h"
#endif

namespace lua_cv {

namespace {
std::string build_preview_html(const PreviewHttpServer::Config& cfg) {
    std::ostringstream oss;
    oss
        << "<!doctype html><html><head><meta charset=\"utf-8\">"
        << "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
        << "<title>LuaScriptVision Preview</title>"
        << "<style>body{font-family:sans-serif;margin:12px;background:#111;color:#eee;}"
        << "#wrap{position:relative;display:inline-block;border:1px solid #444;}"
        << "#player{max-width:100vw;max-height:75vh;background:#000;}"
        << "#overlay{position:absolute;left:0;top:0;pointer-events:none;}"
        << "#bar{margin:8px 0;font-size:14px;} .ok{color:#54d66a}.bad{color:#ff6b6b}"
        << "#json{max-height:24vh;overflow:auto;background:#1a1a1a;padding:8px;border:1px solid #333;}"
        << "</style></head><body>"
        << "<h3>LuaScriptVision Web Preview</h3>"
        << "<div id=\"bar\">"
        << "Video WS(<span id=\"vport\">" << cfg.ws_video_port << "</span>): <span id=\"vst\" class=\"bad\">disconnected</span> | "
        << "Infer WS(<span id=\"iport\">" << cfg.ws_infer_port << "</span>): <span id=\"ist\" class=\"bad\">disconnected</span> | "
        << "FPS: <span id=\"fps\">0</span>"
        << "</div>"
        << "<div id=\"wrap\"><video id=\"player\" autoplay muted playsinline></video><canvas id=\"overlay\"></canvas></div>"
        << "<pre id=\"json\"></pre>"
        << "<script src=\"https://cdn.jsdelivr.net/npm/jmuxer@2.1.0/dist/jmuxer.min.js\"></script>"
        << "<script>(function(){"
        << "const host=location.hostname;"
        << "const vurl=`ws://${host}:" << cfg.ws_video_port << "`;"
        << "const iurl=`ws://${host}:" << cfg.ws_infer_port << "`;"
        << "const video=document.getElementById('player');"
        << "const canvas=document.getElementById('overlay');"
        << "const ctx=canvas.getContext('2d');"
        << "const vst=document.getElementById('vst');"
        << "const ist=document.getElementById('ist');"
        << "const fpsEl=document.getElementById('fps');"
        << "const jsonEl=document.getElementById('json');"
        << "let inferW=" << cfg.infer_width << ", inferH=" << cfg.infer_height << ";"
        << "let last=[]; let frames=0; let lastTs=performance.now();"
        << "const mux=new JMuxer({node:'player',mode:'video',fps:30,flushingTime:0,clearBuffer:true});"
        << "function mark(el,ok){el.textContent=ok?'connected':'disconnected';el.className=ok?'ok':'bad';}"
        << "function resize(){const r=video.getBoundingClientRect();canvas.width=Math.max(1,Math.floor(r.width));canvas.height=Math.max(1,Math.floor(r.height));}"
        << "window.addEventListener('resize',resize); setInterval(resize,500);"
        << "function draw(){ctx.clearRect(0,0,canvas.width,canvas.height);"
        << "const sx=canvas.width/Math.max(1,inferW), sy=canvas.height/Math.max(1,inferH);"
        << "ctx.lineWidth=2; ctx.font='14px sans-serif';"
        << "for(const b of last){const x=(b.x||0)*sx,y=(b.y||0)*sy,w=(b.w||0)*sx,h=(b.h||0)*sy;"
        << "ctx.strokeStyle='#00ff88'; ctx.strokeRect(x,y,w,h);"
        << "const s=(b.score!==undefined)?` ${(b.score*100).toFixed(1)}%`:'';"
        << "const l=(b.label||('cls'+(b.class_id!==undefined?b.class_id:'')))+s;"
        << "ctx.fillStyle='rgba(0,0,0,0.5)'; ctx.fillRect(x,Math.max(0,y-16),Math.max(40,l.length*8),16);"
        << "ctx.fillStyle='#00ff88'; ctx.fillText(l,x+2,Math.max(12,y-4));}"
        << "}"
        << "function parseBoxes(payload){if(Array.isArray(payload)) return payload;"
        << "if(payload&&Array.isArray(payload.boxes)) return payload.boxes;"
        << "if(payload&&Array.isArray(payload.items)){const out=[];for(const it of payload.items){if(Array.isArray(it)) out.push(...it); else if(it&&Array.isArray(it.boxes)) out.push(...it.boxes);}return out;}"
        << "return []; }"
        << "function handleInfer(obj){const d=(obj&&obj.data!==undefined)?obj.data:obj||{};"
        << "if(d.resolution&&Array.isArray(d.resolution)&&d.resolution.length>=2){inferW=Number(d.resolution[0])||inferW;inferH=Number(d.resolution[1])||inferH;}"
        << "if(d.frame_width&&d.frame_height){inferW=Number(d.frame_width)||inferW;inferH=Number(d.frame_height)||inferH;}"
        << "last=parseBoxes(d);"
        << "jsonEl.textContent=JSON.stringify(d,null,2);draw();"
        << "}"
        << "function connectVideo(){const ws=new WebSocket(vurl); ws.binaryType='arraybuffer';"
        << "ws.onopen=()=>mark(vst,true); ws.onclose=()=>{mark(vst,false);setTimeout(connectVideo,1000);}; ws.onerror=()=>mark(vst,false);"
        << "ws.onmessage=(ev)=>{const data=new Uint8Array(ev.data); mux.feed({video:data}); frames++; const now=performance.now(); if(now-lastTs>1000){fpsEl.textContent=String(frames);frames=0;lastTs=now;} resize(); draw();};}"
        << "function connectInfer(){const ws=new WebSocket(iurl);"
        << "ws.onopen=()=>mark(ist,true); ws.onclose=()=>{mark(ist,false);setTimeout(connectInfer,1000);}; ws.onerror=()=>mark(ist,false);"
        << "ws.onmessage=(ev)=>{try{const t=(typeof ev.data==='string')?ev.data:new TextDecoder().decode(new Uint8Array(ev.data)); handleInfer(JSON.parse(t));}catch(e){jsonEl.textContent='infer parse error: '+e;}};}"
        << "connectVideo();connectInfer();resize();"
        << "})();</script></body></html>";
    return oss.str();
}
}  // namespace

#ifdef USE_MONGOOSE_WS
struct PreviewHttpServer::Impl {
    explicit Impl(PreviewHttpServer* owner_ptr)
        : owner(owner_ptr) {
        std::memset(&mgr, 0, sizeof(mgr));
    }

    static void on_event(struct mg_connection* c, int ev, void* ev_data) {
        auto* self = static_cast<Impl*>(c->fn_data);
        if (!self || !self->owner) {
            return;
        }

        if (ev != MG_EV_HTTP_MSG) {
            return;
        }

        auto* hm = static_cast<struct mg_http_message*>(ev_data);
        if (!hm) {
            return;
        }

        const std::string uri(hm->uri.buf, hm->uri.len);
        if (uri == "/" || uri == "/preview" || uri == "/preview.html") {
            mg_http_reply(c,
                          200,
                          "Content-Type: text/html; charset=utf-8\r\nCache-Control: no-store\r\n",
                          "%s",
                          self->html.c_str());
            return;
        }

        if (uri == "/health") {
            mg_http_reply(c, 200, "Content-Type: application/json\r\n", "{\"ok\":true}\n");
            return;
        }

        mg_http_reply(c, 404, "", "Not Found\n");
    }

    PreviewHttpServer* owner = nullptr;
    struct mg_mgr mgr;
    struct mg_connection* listener = nullptr;
    std::string html;
};
#endif

PreviewHttpServer::PreviewHttpServer(const Config& config)
    : config_(config) {
#ifdef USE_MONGOOSE_WS
    impl_ = new Impl(this);
    impl_->html = build_preview_html(config_);
#endif
}

PreviewHttpServer::~PreviewHttpServer() {
    stop();
#ifdef USE_MONGOOSE_WS
    delete impl_;
    impl_ = nullptr;
#endif
}

bool PreviewHttpServer::start() {
    if (running_.load(std::memory_order_acquire)) {
        return true;
    }

#ifndef USE_MONGOOSE_WS
    std::cerr << "[PreviewHTTP] unavailable (USE_MONGOOSE_WS not set)" << std::endl;
    return false;
#else
    if (!impl_) {
        return false;
    }

    mg_mgr_init(&impl_->mgr);
    if (!mg_wakeup_init(&impl_->mgr)) {
        mg_mgr_free(&impl_->mgr);
        return false;
    }

    std::string url = "http://0.0.0.0:" + std::to_string(config_.port);
    impl_->listener = mg_http_listen(&impl_->mgr, url.c_str(), Impl::on_event, impl_);
    if (!impl_->listener) {
        mg_mgr_free(&impl_->mgr);
        std::cerr << "[PreviewHTTP] listen failed on " << url << std::endl;
        return false;
    }

    running_.store(true, std::memory_order_release);
    io_thread_ = std::thread(&PreviewHttpServer::run_loop, this);
    std::cout << "[PreviewHTTP] started at " << get_url() << std::endl;
    return true;
#endif
}

void PreviewHttpServer::stop() {
    if (!running_.exchange(false, std::memory_order_acq_rel)) {
        return;
    }

#ifdef USE_MONGOOSE_WS
    if (impl_) {
        mg_wakeup(&impl_->mgr, impl_->listener ? impl_->listener->id : 1, "x", 1);
    }
#endif

    if (io_thread_.joinable()) {
        io_thread_.join();
    }

#ifdef USE_MONGOOSE_WS
    if (impl_) {
        impl_->listener = nullptr;
        mg_mgr_free(&impl_->mgr);
    }
#endif
}

std::string PreviewHttpServer::get_url(const std::string& host) const {
    return "http://" + host + ":" + std::to_string(config_.port) + "/preview.html";
}

void PreviewHttpServer::run_loop() {
#ifdef USE_MONGOOSE_WS
    while (running_.load(std::memory_order_acquire)) {
        mg_mgr_poll(&impl_->mgr, 20);
    }
#endif
}

}  // namespace lua_cv
