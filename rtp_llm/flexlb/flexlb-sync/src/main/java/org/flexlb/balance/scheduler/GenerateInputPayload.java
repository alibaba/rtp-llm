package org.flexlb.balance.scheduler;

import com.google.protobuf.ByteString;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import com.google.protobuf.InvalidProtocolBufferException;
import com.google.protobuf.UnknownFieldSet;
import org.flexlb.engine.grpc.EngineRpcService.GenerateInputPB;

import java.io.IOException;

/** Validated delivery input with encoded tokens, rather than a decoded token array. */
final class GenerateInputPayload {
    private final GenerateInputPB wireMessage;

    private GenerateInputPayload(GenerateInputPB wireMessage) {
        this.wireMessage = wireMessage;
    }

    static GenerateInputPayload parse(ByteString bytes) throws InvalidProtocolBufferException {
        try {
            CodedInputStream input = bytes.newCodedInput();
            var metadata = GenerateInputPB.newBuilder();
            var tokens = new Tokens();
            int retainedStart = 0;
            int tagStart = 0;
            for (int tag; (tag = input.readTag()) != 0; tagStart = input.getTotalBytesRead()) {
                if (tag == 16 || tag == 18) {
                    mergeMetadata(metadata, bytes, retainedStart, tagStart);
                    if (tag == 18) {
                        tokens.readPacked(input, bytes);
                    } else {
                        tokens.writer().writeInt32NoTag(input.readInt32());
                    }
                    retainedStart = input.getTotalBytesRead();
                } else if (!input.skipField(tag)) {
                    input.checkLastTagWas(0);
                }
            }
            input.checkLastTagWas(0);
            mergeMetadata(metadata, bytes, retainedStart, bytes.size());
            var parsed = metadata.build();
            ByteString encoded = tokens.finish();
            if (!encoded.isEmpty() && bytes.asReadOnlyByteBufferList().size() > 1) {
                // A segmented input may use a copying stream decoder for metadata. Give
                // tokens their own compact storage so they cannot pin copied large fields.
                var compact = ByteString.newOutput(encoded.size());
                encoded.writeTo(compact);
                encoded = compact.toByteString();
            }
            if (!encoded.isEmpty()) {
                // Unknown fields use the standard protobuf writer. On the Engine this is the
                // known packed token_ids field again; no schema or RPC method changes.
                var field = UnknownFieldSet.Field.newBuilder().addLengthDelimited(encoded).build();
                parsed =
                        parsed.toBuilder()
                                .mergeUnknownFields(
                                        UnknownFieldSet.newBuilder().mergeField(2, field).build())
                                .build();
            }
            return new GenerateInputPayload(parsed);
        } catch (InvalidProtocolBufferException malformed) {
            // Diagnostics can outlive dispatch. Never let a parse failure pin a partial
            // tensor/message.
            throw malformed.setUnfinishedMessage(null);
        } catch (IOException failure) {
            throw new InvalidProtocolBufferException(failure);
        }
    }

    /** Wire representation only: token accessors are intentionally empty until Engine decoding. */
    GenerateInputPB.Builder toWireBuilder() {
        return wireMessage.toBuilder();
    }

    private static void mergeMetadata(
            GenerateInputPB.Builder metadata, ByteString bytes, int start, int end)
            throws IOException {
        if (start == end) {
            return;
        }
        var input = bytes.substring(start, end).newCodedInput();
        // ByteString is immutable: tensors and unknown byte fields can share its storage.
        input.enableAliasing(true);
        metadata.mergeFrom(input);
        input.checkLastTagWas(0);
    }

    private static final class Tokens {
        private ByteString packed;
        private ByteString.Output combined;
        private CodedOutputStream writer;

        private void readPacked(CodedInputStream input, ByteString bytes) throws IOException {
            int length = input.readRawVarint32();
            int previousLimit = input.pushLimit(length);
            int start = input.getTotalBytesRead();
            while (input.getBytesUntilLimit() > 0) {
                input.readInt32();
            }
            input.popLimit(previousLimit);
            ByteString chunk = bytes.substring(start, input.getTotalBytesRead());
            if (packed == null && writer == null) {
                packed = chunk;
            } else {
                writer().writeRawBytes(chunk);
            }
        }

        private CodedOutputStream writer() throws IOException {
            if (writer == null) {
                combined = ByteString.newOutput(packed == null ? 32 : Math.max(32, packed.size()));
                writer = CodedOutputStream.newInstance(combined, 64);
                if (packed != null) {
                    writer.writeRawBytes(packed);
                    packed = null;
                }
            }
            return writer;
        }

        private ByteString finish() throws IOException {
            if (writer == null) {
                return packed == null ? ByteString.EMPTY : packed;
            }
            writer.flush();
            return combined.toByteString();
        }
    }
}
