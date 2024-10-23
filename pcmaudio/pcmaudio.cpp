#include "VapourSynth4.h"
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

typedef struct {
    VSAudioInfo ai;
    FILE* pcm_file;
    uint16_t buffer[VS_AUDIO_FRAME_SAMPLES * 2];
} FilterData;


static const VSFrame *VS_CC filterGetFrame(int n, int activationReason, void *instanceData, void **frameData, VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi) {
    FilterData *d = (FilterData *)instanceData;
    auto frmsz = VS_AUDIO_FRAME_SAMPLES * 2 * 2;
    auto bpos = n * frmsz;

    auto ret = fseek(d->pcm_file, bpos, SEEK_SET);
    assert(ret == 0);

    int smplcnt = VS_AUDIO_FRAME_SAMPLES;

    if(n == d->ai.numFrames-1) {
        smplcnt =  d->ai.numSamples - n * VS_AUDIO_FRAME_SAMPLES;
    }

    auto frame = vsapi->newAudioFrame(&d->ai.format, smplcnt,nullptr,core);

    ret = fread(d->buffer, 1, sizeof(uint16_t) * smplcnt * 2, d->pcm_file);
    assert(ret == sizeof(uint16_t) * smplcnt * 2);
    auto ptr0 = reinterpret_cast<uint16_t*>(vsapi->getWritePtr(frame,0));
    auto ptr1 = reinterpret_cast<uint16_t*>(vsapi->getWritePtr(frame,1));
    for (int i = 0; i < smplcnt; i++) {
        ptr0[i] = d->buffer[i * 2 + 0];
        ptr1[i] = d->buffer[i * 2 + 1];
    }
    return frame;
}

static void VS_CC filterFree(void *instanceData, VSCore *core, const VSAPI *vsapi) {
    FilterData *d = (FilterData *)instanceData;
    fclose(d->pcm_file);
}

static void VS_CC filterCreate(const VSMap *in, VSMap *out, void *userData, VSCore *core, const VSAPI *vsapi) {
    FilterData d;
    FilterData *data;

    const char* fpath = vsapi->mapGetData(in, "path", 0, 0);
    d.pcm_file = fopen(fpath, "rb");
    if(!d.pcm_file) {
        vsapi->mapSetError(out,"could not open file");
        return;
    }

    fseek(d.pcm_file, 0L, SEEK_END);
    auto sz = ftell(d.pcm_file);
    auto numSamples = (sz / 2) / 2;

    d.ai = {
        .format = {
            .bitsPerSample = 16,
            .bytesPerSample = 2,
            .numChannels = 2,
            .channelLayout = 3,
        },
        .sampleRate = 44100,
        .numSamples = numSamples,
        .numFrames = static_cast<int>((numSamples + VS_AUDIO_FRAME_SAMPLES - 1) / VS_AUDIO_FRAME_SAMPLES),
    };
    data = (FilterData *)malloc(sizeof(d));
    *data = d;

    vsapi->createAudioFilter(out, "Source", &data->ai, filterGetFrame, filterFree, fmUnordered, NULL, 0, data, core);
}


VS_EXTERNAL_API(void) VapourSynthPluginInit2(VSPlugin *plugin, const VSPLUGINAPI *vspapi) {
    vspapi->configPlugin("com.jsaowji.pcmauidi", "ldpcmaudio", "VapourSynth Filter Skeleton", VS_MAKE_VERSION(1, 0), VAPOURSYNTH_API_VERSION, 0, plugin);
    vspapi->registerFunction("Source", "path:data;", "clip:anode;", filterCreate, NULL, plugin);
}
